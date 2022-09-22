import json
import os
import pickle
import signal
import threading
import time
import zipfile

import gdown
import numpy as np
import requests
import torch
import tqdm
from autocuda import auto_cuda, auto_cuda_name
from findfile import find_files, find_cwd_file, find_file
from termcolor import colored
from functools import wraps

from update_checker import parse_version

from anonymous_demo import __version__


def save_args(config, save_path):
    f = open(os.path.join(save_path), mode='w', encoding='utf8')
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write('{}: {}\n'.format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None, mode=0):
    args = [key for key in sorted(config.args.keys())]
    for arg in args:
        if logger:
            logger.info('{0}:{1}\t-->\tCalling Count:{2}'.format(arg, config.args[arg], config.args_call_count[arg]))
        else:
            print('{0}:{1}\t-->\tCalling Count:{2}'.format(arg, config.args[arg], config.args_call_count[arg]))


def check_and_fix_labels(label_set: set, label_name, all_data, opt):
    if '-100' in label_set:

        label_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_label = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_label' not in opt.args:
        opt.index_to_label = index_to_label
        opt.label_to_index = label_to_index

    if opt.index_to_label != index_to_label:
        opt.index_to_label.update(index_to_label)
        opt.label_to_index.update(label_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = label_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))


def check_and_fix_IOB_labels(label_map, opt):
    index_to_IOB_label = {int(label_map[origin_label]): origin_label for origin_label in label_map}
    opt.index_to_IOB_label = index_to_IOB_label


def get_device(auto_device):
    if isinstance(auto_device, str) and auto_device == 'allcuda':
        device = 'cuda'
    elif isinstance(auto_device, str):
        device = auto_device
    elif isinstance(auto_device, bool):
        device = auto_cuda() if auto_device else 'cpu'
    else:
        device = auto_cuda()
        try:
            torch.device(device)
        except RuntimeError as e:
            print(colored('Device assignment error: {}, redirect to CPU'.format(e), 'red'))
            device = 'cpu'
    device_name = auto_cuda_name()
    return device, device_name


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in tqdm.tqdm(fin.readlines(), postfix='Loading embedding file...'):
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, opt):
    if not os.path.exists('run'):
        os.makedirs('run')
    embed_matrix_path = 'run/{}'.format(os.path.join(opt.dataset_name, dat_fname))
    if os.path.exists(embed_matrix_path):
        print(colored('Loading cached embedding_matrix from {} (Please remove all cached files if there is any problem!)'.format(embed_matrix_path), 'green'))
        embedding_matrix = pickle.load(open(embed_matrix_path, 'rb'))
    else:
        glove_path = prepare_glove840_embedding(embed_matrix_path)
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))

        word_vec = _load_word_vec(glove_path, word2idx=word2idx, embed_dim=embed_dim)

        for word, i in tqdm.tqdm(word2idx.items(), postfix=colored('Building embedding_matrix {}'.format(dat_fname), 'yellow')):
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embed_matrix_path, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class TransformerConnectionError(ValueError):
    def __init__(self):
        pass


def retry(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        count = 5
        while count:

            try:
                return f(*args, **kwargs)
            except (
                TransformerConnectionError,
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ProxyError,
                requests.exceptions.SSLError,
                requests.exceptions.BaseHTTPError,
            ) as e:
                print(colored('Training Exception: {}, will retry later'.format(e)))
                time.sleep(60)
                count -= 1

    return decorated


def save_json(dic, save_path):
    if isinstance(dic, str):
        dic = eval(dic)
    with open(save_path, 'w', encoding='utf-8') as f:
        # f.write(str(dict))
        str_ = json.dumps(dic, ensure_ascii=False)
        f.write(str_)


def load_json(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        print(type(data), data)
        dic = json.loads(data)
    return dic


def init_optimizer(optimizer):
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
        torch.optim.Adadelta: torch.optim.Adadelta,  # default lr=1.0
        torch.optim.Adagrad: torch.optim.Adagrad,  # default lr=0.01
        torch.optim.Adam: torch.optim.Adam,  # default lr=0.001
        torch.optim.Adamax: torch.optim.Adamax,  # default lr=0.002
        torch.optim.ASGD: torch.optim.ASGD,  # default lr=0.01
        torch.optim.RMSprop: torch.optim.RMSprop,  # default lr=0.01
        torch.optim.SGD: torch.optim.SGD,
        torch.optim.AdamW: torch.optim.AdamW,
    }
    if optimizer in optimizers:
        return optimizers[optimizer]
    elif hasattr(torch.optim, optimizer.__name__):
        return optimizer
    else:
        raise KeyError('Unsupported optimizer: {}. Please use string or the optimizer objects in torch.optim as your optimizer'.format(optimizer))
