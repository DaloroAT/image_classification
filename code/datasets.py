from typing import Tuple, List, Optional, Sequence, Union
import torch

from torch.utils.data.dataset import Dataset
from pandas import DataFrame
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as t
from functools import lru_cache

SIZE = (64, 64)


class TinyImagenetDataset(Dataset):
    _df: DataFrame
    _transforms: t.Compose

    def __init__(self, dataframe: DataFrame):
        self._df = dataframe
        self._transform = None

        self.set_transforms(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Path]:
        path, label = self._df.loc[idx, :]
        image = _get_image(path)
        image = self._transform(image)
        return image, label, path

    def set_transforms(self, aug_degree: float) -> None:
        if aug_degree == 0:
            transforms = basic_transforms()
        else:
            transforms = t.Compose([random_transforms(aug_degree),
                                    basic_transforms()])
        self._transform = transforms

    def __len__(self) -> int:
        return len(self._df)


@lru_cache(maxsize=2**17)
def _get_image(path: Path) -> Image:
    image = Image.open(path).convert("RGB")
    return image


def basic_transforms() -> t.Compose:
    transforms = t.Compose([t.ToTensor(),
                            t.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])
    return transforms


def random_transforms(aug_degree: float) -> t.RandomOrder:
    assert 0 <= aug_degree <= 3

    aug_list = [
        t.functional.hflip,
        t.RandomResizedCrop(size=SIZE, scale=(7/8, 7/8)),
        t.RandomAffine(degrees=aug_degree * 10,
                       translate=(0.1 * aug_degree, 0.1 * aug_degree),
                       scale=(1 - 0.1 * aug_degree, 1 + 0.1 * aug_degree),
                       shear=aug_degree * 5,
                       fillcolor=0
                       ),
        t.ColorJitter(brightness=0.1 * aug_degree,
                      contrast=0.1 * aug_degree,
                      saturation=0.1 * aug_degree
                      )
    ]
    transforms = t.RandomOrder([t.RandomApply([aug], p=0.4 + 0.1 * aug_degree) for aug in aug_list])
    return transforms
