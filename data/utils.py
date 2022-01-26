"""Utilites for making and modifying dataframe."""
import os
from typing import Dict, Union, List

import pandas as pd  # type:ignore
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import LabelEncoder  # type: ignore
import sklearn  # type: ignore


def get_landmark_ids(dataframe: pd.DataFrame, num_images: int) -> Dict[str, List[int]]:
    """Extract landmark_id(labels) that we are going to use in our pipeline
         since we are not going to use all training images.

    Parameters
    -----------

    dataframe: pd.DataFrame
        Training csv that contains landmark ids

    num_images: int
        Minimum number of images which class contains, for example if num_images=200,
         it will return ids that have atleast 200 images.

    Returns
        used_landmarks: list
            List of  landmark_ids that have more images than num_images.

        not_used_landmarks: list
            List of landmark_ids that have less images than num_images.
    """
    count: pd.Series = dataframe["landmark_id"].value_counts()
    used_landmarks = count[count >= num_images].index.tolist()
    not_used_landmarks = count[count < num_images].index.tolist()

    return {"used_landmarks": used_landmarks, "not_used_landmarks": not_used_landmarks}


def get_image_fpaths(dataframe: pd.DataFrame, train: bool) -> pd.Series:
    """Get image file paths as a dataframe column for reading images.
    Image file path is id[0] + id[1] + id[2] + id + .jpg.I will be storing data
    on S3 Bucket.

    Parameters
    ----------

    dataframe: pd.DataFrame
        Training csv which contains image ids

    train: bool
        If true it will make paths for training images which means DIR will be train/ else test/

    Returns
        image_fpaths: pd.Series
            Pandas dataframe column containg image file paths
    """
    if train:
        sub_dir = "train"
    else:
        sub_dir = "test"

    image_fpaths = dataframe["id"].map(
        lambda image_id: os.path.join(
            sub_dir, image_id[0], image_id[1], image_id[2], image_id + ".jpg"
        )
    )

    return image_fpaths


def label_encoder(
    dataframe: pd.DataFrame, target_column: str = "landmark_id"
) -> Dict[str, Union[sklearn.preprocessing.LabelEncoder, npt.NDArray[np.int64]]]:
    """Label encode target column from dataframe.

    Parameters
    ----------

    dataframe: pd.DataFrame
        Training csv that contains target column

    target_col: str
        Name of target column

    Returns
        la: sklearn.preprocessing.LabelEncoder
            label encoder class

        encoded_target: np.array
            arrays of encoded values for target class from 0 to num_classes - 1
    """
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(dataframe[target_column])

    return {
        "encoder": encoder,
        "encoded_target": encoded_target,
    }
