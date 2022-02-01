import os

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing as skp
from dotenv import load_dotenv

from data.utils import get_image_fpaths, get_landmark_ids, label_encoder

load_dotenv()  # read env vars

df: pd.DataFrame = pd.read_csv(os.environ.get("TRAIN_CSV"))


@pytest.mark.parametrize("num_images", [200, 300, 400])
def test_get_landmarks(num_images: int) -> None:
    "Test type of output, number of landmarks"

    num_landmarks = df["landmark_id"].nunique()
    landmarks = get_landmark_ids(dataframe=df, num_images=num_images)
    count: pd.Series = df["landmark_id"].value_counts()
    used_landmarks = count[count >= num_images].index.tolist()
    not_used_landmarks = count[count < num_images].index.tolist()

    assert len(landmarks) == 2
    assert "used_landmarks" in landmarks and "not_used_landmarks" in landmarks
    assert isinstance(landmarks, dict)
    assert isinstance(landmarks["used_landmarks"], list)
    assert isinstance(landmarks["not_used_landmarks"], list)
    assert (
        len(landmarks["used_landmarks"]) + len(landmarks["not_used_landmarks"])
        == num_landmarks
    )
    assert used_landmarks == landmarks["used_landmarks"]
    assert not_used_landmarks == landmarks["not_used_landmarks"]


@pytest.mark.parametrize("train", [True, False])
def test_image_fpaths(train: bool) -> None:
    """Test length, test that every image ends with .jpg,
    test that ever training image starts with train/ DIR
    and test image with test/ DIR
    """

    image_fpaths = get_image_fpaths(dataframe=df, train=train)

    assert isinstance(image_fpaths, pd.Series)
    assert len(image_fpaths) == len(df)
    assert all([fpath.endswith(".jpg") for fpath in image_fpaths])

    if train:
        assert all(["train" in fpath for fpath in image_fpaths])
    else:
        assert all(["test" in fpath for fpath in image_fpaths])


def test_label_encoder() -> None:
    "Test that values are from 0 to num_classes -1 and length of encoded class"

    out = label_encoder(dataframe=df)
    encoder = out["encoder"]
    encoded_column = out["encoded_target"]

    assert isinstance(out, dict)
    assert isinstance(encoder, skp.LabelEncoder)
    assert isinstance(encoded_column, np.ndarray)
    assert list(encoder.classes_) == df["landmark_id"].unique().tolist()
    assert min(encoded_column) == 0
    assert max(encoded_column) == df["landmark_id"].nunique() - 1
