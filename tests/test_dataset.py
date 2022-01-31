# type: ignore
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import pytest

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
    assert "images" in dataset[1000] and "labels" in dataset[10000]
    if transform is None:
        assert isinstance(dataset[0]["images"], np.ndarray)
    if transform is AUGMENTS:
        assert isinstance(dataset[100]["images"], torch.Tensor)
