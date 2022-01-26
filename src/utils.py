"""Utilities for working with s3 and local files"""
import io
import sys
import pickle
import json
from typing import Any

import yaml  # type: ignore
import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from boto3_type_annotations.s3 import Client  # type: ignore
import tqdm  # type: ignore


def artifact_to_s3(
    object_: Any, bucket: str, key: str, extension: str = "json", verbose: bool = True
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
    file_like_object: io.BytesIO, key: str, bucket: str, extension: str, verbose: bool
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
                    Callback=lambda bytes_transfered: pbar.update(bytes_transfered),
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
