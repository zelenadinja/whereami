"""Data augmentations"""

from typing import Callable

import albumentations
from albumentations.pytorch import ToTensorV2


def aug_version_0(config: dict) -> Callable:
    """augmentations for version 0,just resize"""

    return albumentations.Compose(
        [
            albumentations.RandomCrop(config['size'], config['size']),
            albumentations.Normalize(config['mean'], config['std']),
            ToTensorV2(),
        ]
    )


def aug_version_1(config: dict, train: bool) -> Callable:
    """augments for version 1"""
    if train:
        return albumentations.Compose(
            [
                albumentations.SmallestMaxSize(
                    config['train_small_max_size']
                ),
                albumentations.RandomCrop(
                    config['train_crop'], config['train_crop']
                ),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Normalize(config['mean'], config['std']),
                ToTensorV2(),

            ]
        )
    return albumentations.Compose(
        [
            albumentations.SmallestMaxSize(config['valid_small_max_size']),
            albumentations.CenterCrop(
                config['valid_crop'], config['valid_crop']
            ),
            albumentations.Normalize(config['mean'], config['std']),
            ToTensorV2(),
        ]
    )
