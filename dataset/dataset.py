"""Dataset for reading images"""

from typing import Dict, Union, Callable, Optional

import torch
from torch.utils.data import Dataset
import numpy.typing as npt
import pandas as pd
import numpy as np

from src.utils import read_image_s3


class LandmarkDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Callable[[npt.NDArray[np.float32]], torch.Tensor]]) -> None:

        self.image_paths = np.array(dataframe['object_key'])
        self.targets = np.array(dataframe['target'])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, x: int) -> Dict[str, Union[torch.Tensor, npt.NDArray[np.float32]]]:

        image_path = self.image_paths[x]
        image = read_image_s3(object_key=image_path)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.targets[x], dtype=torch.long)

        return {
            'images': image,
            'labels': label
        }



