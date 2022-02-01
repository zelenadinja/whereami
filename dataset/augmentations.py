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
