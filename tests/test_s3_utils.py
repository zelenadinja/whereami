import os
import io

import boto3
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import pytest
import timm
import torch

from src.utils import artifact_to_s3, read_image_s3, read_artifacts_s3
from models.utils import save_checkpoint_to_s3

load_dotenv()  # envs
S3_BUCKET: str = os.environ.get('S3_BUCKET')

@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("extension", ["json", "yaml", "pkl"])
def test_artifacts_to_s3(verbose: bool, extension: "str") -> None:
    "Test if artifact is uplaoded for all extensions, test for wrong extension"

    sample_list = [1, 2, 3, 4, 5]
    artifact_to_s3(
        object_=sample_list,
        bucket=S3_BUCKET,
        key="test_artifacts/test_file",
        extension=extension,
        verbose=verbose,
    )
    json_objects = []
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(S3_BUCKET)  # pylint: disable=maybe-no-member
    for i in bucket.objects.filter(Prefix="test_artifacts"):
        json_objects.append(i.key)

    if extension == "json":
        assert "test_artifacts/test_file.json" in json_objects
    elif extension == "pkl":
        assert "test_artifacts/test_file.pkl" in json_objects
    elif extension == "yaml":
        assert "test_artifacts/test_file.yaml" in json_objects

    with pytest.raises(ValueError):
        artifact_to_s3(
            object_=sample_list,
            bucket=S3_BUCKET,
            key="test",
            extension="csv",
            verbose=True,
        )

def test_reading_images() -> None:
    """Testing reading images from S3 Bucket, sample out 30 images images and
       check their data type."""

    dataframe = pd.read_csv(os.environ.get('PROCESSED_TRAIN_CSV'))
    object_keys = np.array(dataframe['object_key'])
    random_keys = np.random.choice(object_keys, size=30)

    for i in random_keys:
        img = read_image_s3(object_key=i)
        assert isinstance(img, np.ndarray)

    with pytest.raises(ValueError):
        read_image_s3(object_key='nonexistingimage')


def test_reading_artifacts() -> None:
    """Testin reading artifacts from S3 Bucket"""

    sample_obj = [1, 2, 3, 4, 5]
    artifact_to_s3(object_=sample_obj, bucket=S3_BUCKET, key='sample', verbose=False)

    assert sample_obj == read_artifacts_s3(object_key='sample.json')

def test_saving_checkpoints() -> None:
    """Test saving checkpoints to s3 bucket"""
    model = timm.create_model('resnet50', pretrained=True)
    checkpoint = {
        'model':model.state_dict()
    }
    save_checkpoint_to_s3(checkpoint=checkpoint, checkpoint_name='test_checkpoint')
    s3_resource = boto3.resource('s3')
    checkpoint_obj = s3_resource.Object(S3_BUCKET, 'checkpoints/test_checkpoint.pth')
    checkpoint_body = checkpoint_obj.get()['Body'].read()
    checkpoint = torch.load(io.BytesIO(checkpoint_body))

    assert (checkpoint['model']['conv1.weight'] == model.state_dict()['conv1.weight']).all().item()





