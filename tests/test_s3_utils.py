"""Testing utilities for S3 Bucket"""

import os

import boto3
from dotenv import load_dotenv
import pytest

from src.utils import artifact_to_s3


load_dotenv()  # envs
S3_BUCKET = os.environ.get('S3_BUCKET')

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
