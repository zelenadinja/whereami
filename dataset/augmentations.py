"""Data augmentations"""

from typing import Callable

import albumentations
from albumentations.pytorch import ToTensorV2


def aug_version_0(config: dict) -> Callable:
    """augmentations for version 0, same for both train and valid,
    just resize and normalize"""

    return albumentations.Compose(
        [
            albumentations.Resize(config['size'], config['size']),
            albumentations.Normalize(),  # imagnet
            ToTensorV2(),
        ]
    )


def aug_version_1(config: dict, train: bool) -> Callable:
    """augmentations for version 1"""

    if train:
        return albumentations.Compose(
            [
                albumentations.Resize(
                    config['train_size'], config['train_size'],
                ),
                albumentations.RandomResizedCrop(
                    config['crop_size'], config['crop_size']
                ),
                albumentations.Normalize(),
                albumentations.HorizontalFlip(),
                albumentations.Cutout(),
                albumentations.RandomBrightnessContrast(),
                albumentations.ShiftScaleRotate(),
                ToTensorV2(),
            ]
        )

    return albumentations.Compose(
        [
            albumentations.Resize(config['valid_size'], config['valid_size']),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )


def aug_version_2(config: dict, train: bool) -> Callable:
    """augmentations for version 2"""
    if train:
        return albumentations.Compose(
            [
                albumentations.Resize(
                    config['train_size'], config['train_size'],
                ),
                albumentations.RandomResizedCrop(
                    config['crop_size'], config['crop_size']
                ),
                albumentations.HorizontalFlip(),
                albumentations.Normalize(),
                ToTensorV2(),
            ]
        )
    return albumentations.Compose(
        [
            albumentations.Resize(config['valid_size'], config['valid_size']),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )
