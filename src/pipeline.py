from __future__ import annotations

import albumentations as A
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import DentalSegDataset, subset_from_split
from .sampler import WeightedSourceBatchSampler


def train_transform():
    return A.Compose([A.HorizontalFlip(p=0.5), A.Affine(scale=(0.9, 1.1), rotate=(-7, 7), p=0.5),
                      A.RandomBrightnessContrast(p=0.3)])


def make_dataset(config: dict, splits: dict, name: str, augment: bool = False) -> DentalSegDataset:
    frame = subset_from_split(config["data"]["manifest"], splits, name)
    if frame.empty:
        raise ValueError(f"'{name}' bölümü boş.")
    return DentalSegDataset(frame, train_transform() if augment else None)


def make_train_loader(dataset: DentalSegDataset, config: dict):
    training = config["training"]
    try:
        sampler = WeightedSourceBatchSampler(dataset.source_labels, training["batch_size"],
                                             training.get("pano_ratio", 0.75), seed=config["seed"])
        return DataLoader(dataset, batch_sampler=sampler, num_workers=config["data"].get("num_workers", 0))
    except ValueError:
        return DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                          num_workers=config["data"].get("num_workers", 0))


def make_eval_loader(dataset: DentalSegDataset, config: dict):
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                      num_workers=config["data"].get("num_workers", 0))

