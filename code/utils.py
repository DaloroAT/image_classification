import random
import os

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from typing import List, Tuple, Dict


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataframes(data_path: Path) -> Tuple[List[DataFrame], Dict]:
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"{data_path} is not a directory.")

    labels_num2txt = get_labels_num2txt(data_path)
    data_path = data_path / 'tiny-imagenet-200'

    all_folders = [
        dir_name
        for _, d, _ in os.walk(str(data_path / 'train'))
        for dir_name in d
        if dir_name != "images"
    ]

    folders_to_num = {val: index for index, val in enumerate(all_folders)}

    val_labels = pd.read_csv(
        data_path / "val" / "val_annotations.txt", sep="\t", header=None, index_col=0)[1].to_dict()

    dataframes = []

    for mode in ['train', 'val']:
        all_files = [
            os.path.join(r, file)
            for r, _, f in os.walk(str(data_path / mode))
            for file in f
            if ".JPEG" in file
        ]
        if mode == 'train':
            labels = [folders_to_num[os.path.basename(file).split("_")[0]] for file in all_files]
        else:
            labels = [folders_to_num[val_labels[os.path.basename(file)]] for file in all_files]

        dataframes.append(pd.DataFrame({"path": all_files, "label": labels}))

    return dataframes, labels_num2txt


def get_labels_num2txt(data_path: Path) -> Dict:
    data_path = data_path / 'tiny-imagenet-200'
    all_folders = [
        dir_name
        for _, d, _ in os.walk(str(data_path / 'train'))
        for dir_name in d
        if dir_name != "images"
    ]

    folders_to_txt = pd.read_csv(
        data_path / "words.txt", sep="\t", header=None, index_col=0)[1].to_dict()

    num_to_folders = {index: val for index, val in enumerate(all_folders)}
    labels_txt = [folders_to_txt[num_to_folders[x]] for x in num_to_folders.keys()]
    labels_num2txt = dict(zip(num_to_folders.keys(), labels_txt))
    return labels_num2txt


class AvgMoving:
    n: int
    avg: float

    def __init__(self):
        self.n = 0
        self.avg = 0

    def add(self, val: float) -> None:
        self.n += 1
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg

    def clear(self) -> None:
        self.n = 0
        self.avg = 0


class Stopper:
    max_wrongs: int
    n_obs_wrongs: int
    delta: float
    best_value: float

    def __init__(self, max_wrongs: int, delta: float):
        assert max_wrongs > 1 and delta > 0
        self.max_wrongs = max_wrongs
        self.n_obs_wrongs = 0
        self.delta = delta
        self.best_value = 0

    def update(self, new_value: float) -> None:
        if new_value - self.best_value < self.delta or new_value < self.best_value:
            self.n_obs_wrongs += 1
        else:
            self.n_obs_wrongs = 0
            self.best_value = new_value

    def is_need_stop(self) -> bool:
        return self.n_obs_wrongs >= self.max_wrongs


def parse_var(s: str) -> Tuple[str, str]:
    # Parse a key, value pair, separated by '='
    items = s.split('=')
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)


def parse_to_dict(items: List[str]) -> Dict:
    # Parse a series of key-value pairs and return a dictionary
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = eval(value)
    return d


def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()

    scores = model(x)
    scores = torch.gather(scores, dim=1, index=y.unsqueeze(1))

    ones_tensor = torch.ones(scores.size())
    scores.backward(gradient=ones_tensor)

    saliency = x.grad

    # Convert 3d to 1d
    saliency = saliency.abs()
    saliency, _ = torch.max(saliency, dim=1)

    return saliency
