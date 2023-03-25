import random
from difflib import Differ

from textattack.attack_recipes import BAEGarg2019
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from findfile import find_files
from flask import Flask
from textattack import Attacker


class ModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model):
        self.model = model  # pipeline = pipeline

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.infer(text_input, print_result=False, **kwargs)
            outputs.append(raw_outputs["probs"])
        return outputs


class SentAttacker:
    def __init__(self, model, recipe_class=BAEGarg2019):
        model = model
        model_wrapper = ModelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')

        # recipe.transformation.language = "en"

        _dataset = [("", 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


def get_ensembled_tad_results(results):
    target_dict = {}
    for r in results:
        target_dict[r["label"]] = (
            target_dict.get(r["label"]) + 1 if r["label"] in target_dict else 1
        )

    return dict(zip(target_dict.values(), target_dict.keys()))[
        max(target_dict.values())
    ]



def get_sst2_example():
    filter_key_words = [
        ".py",
        ".md",
        "readme",
        "log",
        "result",
        "zip",
        ".state_dict",
        ".model",
        ".png",
        "acc_",
        "f1_",
        ".origin",
        ".adv",
        ".csv",
    ]

    dataset_file = {"train": [], "test": [], "valid": []}
    dataset = "sst2"
    search_path = "./"
    task = "text_defense"
    dataset_file["test"] += find_files(
        search_path,
        [dataset, "test", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train."]
                    + filter_key_words,
    )

    for dat_type in ["test"]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:
            with open(data_file, mode="r", encoding="utf8") as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split("$LABEL$")
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def get_agnews_example():
    filter_key_words = [
        ".py",
        ".md",
        "readme",
        "log",
        "result",
        "zip",
        ".state_dict",
        ".model",
        ".png",
        "acc_",
        "f1_",
        ".origin",
        ".adv",
        ".csv",
    ]

    dataset_file = {"train": [], "test": [], "valid": []}
    dataset = "agnews"
    search_path = "./"
    task = "text_defense"
    dataset_file["test"] += find_files(
        search_path,
        [dataset, "test", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train."]
                    + filter_key_words,
    )
    for dat_type in ["test"]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:
            with open(data_file, mode="r", encoding="utf8") as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split("$LABEL$")
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def get_amazon_example():
    filter_key_words = [
        ".py",
        ".md",
        "readme",
        "log",
        "result",
        "zip",
        ".state_dict",
        ".model",
        ".png",
        "acc_",
        "f1_",
        ".origin",
        ".adv",
        ".csv",
    ]

    dataset_file = {"train": [], "test": [], "valid": []}
    dataset = "amazon"
    search_path = "./"
    task = "text_defense"
    dataset_file["test"] += find_files(
        search_path,
        [dataset, "test", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train."]
                    + filter_key_words,
    )

    for dat_type in ["test"]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:
            with open(data_file, mode="r", encoding="utf8") as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split("$LABEL$")
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]


def get_imdb_example():
    filter_key_words = [
        ".py",
        ".md",
        "readme",
        "log",
        "result",
        "zip",
        ".state_dict",
        ".model",
        ".png",
        "acc_",
        "f1_",
        ".origin",
        ".adv",
        ".csv",
    ]

    dataset_file = {"train": [], "test": [], "valid": []}
    dataset = "imdb"
    search_path = "./"
    task = "text_defense"
    dataset_file["test"] += find_files(
        search_path,
        [dataset, "test", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train."]
                    + filter_key_words,
    )

    for dat_type in ["test"]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:
            with open(data_file, mode="r", encoding="utf8") as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split("$LABEL$")
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)
        return data[random.randint(0, len(data))]

