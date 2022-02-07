"""Dataset for reading images"""

from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
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

    def __getitem__(
        self, item_index: int
    ) -> Tuple[Union[Tensor, np.ndarray], Tensor]:

        image_path = self.image_paths[item_index]
        image = read_image_s3(
            object_key=image_path,
            bucket_name='landmarkdataset'
        )

        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.targets[item_index], dtype=torch.long)

        return image, label
