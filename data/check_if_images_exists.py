"""Test if all images are on s3"""

if __name__ == "__main__":

    import math
    import os

    import boto3
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    DF = pd.read_csv(os.environ["PROCESSED_TRAIN_CSV"])
    DF_OBJECT_KEYS = np.array(DF["object_key"])
    S3_BUCKET = os.environ['S3_BUCKET']
    S3CLIENT = boto3.client("s3")
    S3_OBJECT_KEYS = []
    OBJS = [
        obj["Key"]
        for obj in S3CLIENT.list_objects_v2(
            Bucket=S3_BUCKET, Prefix="train"
            )["Contents"]
    ]
    S3_OBJECT_KEYS.append(OBJS)
    DIVIDER = math.floor(len(DF_OBJECT_KEYS) / 1000)

    for i in range(DIVIDER):
        objs = [
            obj["Key"]
            for obj in S3CLIENT.list_objects_v2(
                Bucket="landmarkdataset",
                StartAfter=S3_OBJECT_KEYS[-1][-1],
                Prefix="train",
            )["Contents"]
        ]
        S3_OBJECT_KEYS.append(objs)

    S3_OBJECT_KEYS = list(np.concatenate(S3_OBJECT_KEYS))
    S3_OBJECT_KEYS.sort()
    DF_OBJECT_KEYS.sort()

    if (DF_OBJECT_KEYS == S3_OBJECT_KEYS).all():
        print("ALL KEYS MATCHES")
    else:
        print("Keys from dataframe does not match keys on Bucket")
