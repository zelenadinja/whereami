"""Utilities for networks"""
import io
import os
from typing import Optional, Dict

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import tqdm
import torch
from boto3_type_annotations.s3 import Client


load_dotenv()


def load_weights_from_s3(model_name: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load weights from S3 Bucket"""

    bucket = os.environ.get("S3_BUCKET")
    s3_client: Client = boto3.client("s3")
    try:
        filesize = int(
            s3_client.head_object(
                Bucket="landmarkdataset", Key=f"pretrainedweights/{model_name}.pth"
            )["ResponseMetadata"]["HTTPHeaders"]["content-length"]
        )
    except ClientError:
        raise ValueError(f'Weights for {model_name} does not exist on S3 Bucket.')

    try:
        buffer = io.BytesIO()
        with tqdm.tqdm(
                total=filesize,
                unit="B",
                unit_scale=True,
                ascii=True,
                desc="Loading weights from S3",
        ) as pbar:
            s3_client.download_fileobj(
                Bucket=bucket,
                Key=f"pretrainedweights/{model_name}.pth",
                Fileobj=buffer,
                Callback=lambda bytes_transfered: pbar.update(bytes_transfered),
            )
        buffer.seek(0)  # read buffer from beginning
        weights = torch.load(buffer)
    except ClientError:
        return None
    return weights
