import io
import sys
import os
import pickle
import json
from typing import Any
import random

import yaml
import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv
import boto3
from PIL import Image
from botocore.exceptions import ClientError
from boto3_type_annotations.s3 import Client, ServiceResource
import tqdm
import torch


load_dotenv()


class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all values to 0"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, num: int = 1) -> None:
        """Update values by given val and n"""
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def artifact_to_s3(
        object_: Any, bucket: str, key: str,
        extension: str = "json", verbose: bool = True
) -> bool:
    """Uploads object on S3 Bucket.

    Parameters
    ----------

    object: Any
        Object to be uplaoded.Such as lists, dicts, tuples or label encoder.

    bucket: str
        S3 Bucket name.

    extension: str
        Extension of file

    key: str
        Name of object key for S3 Bucket.

    verbose: bool
        If True progress bar is shown

    Returns
        True if uplaoded else False
    """

    if extension not in ("json", "yaml", "pkl"):
        raise ValueError("Object extensions must be json or yaml")

    if extension == "json":
        streaming_object = io.BytesIO(
            json.dumps(object_).encode()
        )  # Make json object, encode it since BytesIO expects bytes
    elif extension == "yaml":
        streaming_object = io.BytesIO(
            yaml.safe_dump(object_).encode()
        )  # Same with yaml
    elif extension == "pkl":
        streaming_object = io.BytesIO(
            pickle.dumps(object_)
        )  # Mostly for objects like label encoder

    return _upload_file_obj(
        file_like_object=streaming_object,
        bucket=bucket,
        key=key,
        extension=extension,
        verbose=verbose,
    )


def _upload_file_obj(
        file_like_object: io.BytesIO, key: str,
        bucket: str, extension: str, verbose: bool
) -> bool:

    filesize: int = sys.getsizeof(file_like_object)
    s3client: Client = boto3.client("s3")
    try:
        if verbose:
            with tqdm.tqdm(
                    total=filesize,
                    unit="B",
                    unit_scale=True,
                    ascii=True,
                    desc=f"Uplaoding {key} to S3 Bucket",
            ) as pbar:
                s3client.upload_fileobj(
                    Fileobj=file_like_object,
                    Bucket=bucket,
                    Key=key + "." + extension,
                    Callback=lambda bytes_: pbar.update(bytes_),
                    )
        else:
            s3client.upload_fileobj(
                Fileobj=file_like_object,
                Bucket=bucket,
                Key=key + "." + extension,
            )
    except ClientError:
        return False
    return True


def read_image_s3(object_key: str) -> npt.NDArray[np.uint8]:
    """Read image from S3 Bucket into numpy array
    Parameters
    ----------

    object_key: str
        Object key on S3 Bucket with all sub-DIRs
         example: train/0/0/0/023132101.jpg

    Returns:
        np_image: np.ndarray
            image as a numpy array
    """

    s3_resource: ServiceResource = boto3.resource('s3')
    bucket_name = os.environ.get('S3_BUCKET')
    try:
        image_body = s3_resource.Object(bucket_name, object_key).get()['Body']
        image = Image.open(image_body)
        np_image = np.array(image)
    except ClientError:
        raise ValueError('Object key does not exist!')
    return np_image


def read_artifacts_s3(object_key: str) -> Any:
    """Read json,yaml or pickle files from S3 Bucket

    Parameters
    ----------

    object_key: str
        Object key of item, include SUB-DIRs if exists

    Returns
        object:

    """
    s3_resource: ServiceResource = boto3.resource('s3')
    bucket = os.environ.get('S3_BUCKET')
    try:
        obj = s3_resource.Object(bucket, object_key)
    except ClientError:
        raise ValueError('Object key does not exists')

    _, extension = os.path.splitext(object_key)

    if extension == '.json':
        artifact = json.loads(obj.get()['Body'].read())
    elif extension == '.yaml':
        artifact = yaml.safe_load(obj.get()['Body'])
    elif extension == '.pkl':
        artifact = pickle.loads(obj.get()['Body'].read())
    else:
        raise ValueError('Supports yaml,json and pkl format')

    return artifact


def set_seed(seed: int = 42) -> None:
    """REPRODUCIBILITY"""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
