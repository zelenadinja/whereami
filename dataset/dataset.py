"""Dataset for reading images"""

from typing import Callable

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import read_image_s3


class LandmarkDataset(Dataset):
    """Pytorch dataset class for landmark dataframe"""

    def __init__(self, dataframe: pd.DataFrame, transform: Callable) -> None:

        self.image_paths = np.array(dataframe['object_key'])
        self.targets = np.array(dataframe['target'])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item_index: int) -> dict:

        image_path = self.image_paths_kaggle[item_index]
        image = read_image_s3(object_key=image_path)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.targets[item_index], dtype=torch.long)

        return {
            'images': image,
            'labels': label
        }



