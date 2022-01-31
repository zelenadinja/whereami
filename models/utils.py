import io
import os
from typing import Optional
import sys

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import tqdm
import torch
from boto3_type_annotations.s3 import Client


load_dotenv()


def load_weights_from_s3(model_name: str) -> dict:
    """Load weights from S3 Bucket"""

    bucket = os.environ.get("S3_BUCKET")
    s3_client: Client = boto3.client("s3")
    try:
        filesize = int(
            s3_client.head_object(
                Bucket="landmarkdataset",
                Key=f"pretrainedweights/{model_name}.pth"
            )["ResponseMetadata"]["HTTPHeaders"]["content-length"]
        )
    except ClientError:
        raise ValueError(
            f"Weights for {model_name} does not exist on S3 Bucket."
            )

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
                Callback=lambda bytes_: pbar.update(bytes_),
            )
        buffer.seek(0)  # read buffer from beginning
        weights = torch.load(buffer)
    except ClientError:
        raise ValueError("Could not load weights")
    return weights


def save_checkpoint_to_s3(
    checkpoint: dict, checkpoint_name: str
) -> Optional[bool]:
    """Save training checkpoint directly to S3 Bucket"""

    s3client: Client = boto3.client("s3")
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    filesize = sys.getsizeof(buffer)
    try:
        with tqdm.tqdm(
                total=filesize,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Saving checkpoint to S3",
        ) as pbar:
            s3client.upload_fileobj(
                Fileobj=buffer,
                Bucket=os.environ.get("S3_BUCKET"),
                Key=f"checkpoints/{checkpoint_name}.pth",
                Callback=lambda bytes_transfed: pbar.update(bytes_transfed),
            )
    except ClientError:
        return None
    return True
