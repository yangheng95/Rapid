import os
from findfile import find_files, find_dir

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
    ".backup",
    ".bak",
]


def detect_infer_dataset(dataset_path, task="apc"):
    dataset_file = []
    if isinstance(dataset_path, str) and os.path.isfile(dataset_path):
        dataset_file.append(dataset_path)
        return dataset_file

    for d in dataset_path:
        if not os.path.exists(d):
            search_path = find_dir(
                os.getcwd(),
                [d, task, "dataset"],
                exclude_key=filter_key_words,
                disable_alert=False,
            )
            dataset_file += find_files(
                search_path,
                [".inference", d],
                exclude_key=["train."] + filter_key_words,
            )
        else:
            dataset_file += find_files(
                d, [".inference", task], exclude_key=["train."] + filter_key_words
            )

    return dataset_file
