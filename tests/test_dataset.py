#  type: ignore
import os
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

from dataset.dataset import LandmarkDataset

load_dotenv()

AUGMENTS = A.Compose(
    [
        A.Resize(300, 300),
        A.Cutout(),
        A.Normalize(),
        ToTensorV2(),
    ]
)


@pytest.mark.parametrize("transform", [AUGMENTS, None])
def test_pytorch_dataset(transform: Any) -> None:
    """Testing reading images and data type of output"""

    dataframe = pd.read_csv(os.environ.get("PROCESSED_TRAIN_CSV"))
    dataset = LandmarkDataset(dataframe=dataframe, transform=transform)

    assert len(dataset) == len(dataframe)
    if transform is None:
        assert isinstance(dataset[0][0], np.ndarray)
    if transform is AUGMENTS:
        assert isinstance(dataset[100][1], torch.Tensor)
