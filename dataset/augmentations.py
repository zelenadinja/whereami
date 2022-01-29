"""Data augmentations"""

from typing import Callable

import albumentations
from albumentations.pytorch import ToTensorV2

def aug_version_1(config: dict) -> Callable:
    """augmentations for version 1, same for both train and valid,
    just resize and normalize"""

    return albumentations.Compose(
        [
            albumentations.Resize(config['size'], config['size']),
            albumentations.Normalize(), #imagnet
            ToTensorV2(),
        ]
    )
