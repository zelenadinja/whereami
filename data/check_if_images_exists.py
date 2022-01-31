"""Test if all images are on s3"""

if __name__ == "__main__":

    import os
    import math

    from dotenv import load_dotenv
    import numpy as np
    import pandas as pd
    import boto3

    load_dotenv()

    DF = pd.read_csv(os.environ.get("PROCESSED_TRAIN_CSV"))
    DF_OBJECT_KEYS = np.array(DF["object_key"])

    S3CLIENT = boto3.client("s3")
    S3_OBJECT_KEYS = []
    OBJS = [
        obj["Key"]
        for obj in S3CLIENT.list_objects_v2(
            Bucket="landmarkdataset", Prefix="train"
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
    S3_OBJECT_KEYS.remove("train/dataset_lock.lock")
    S3_OBJECT_KEYS.sort()
    DF_OBJECT_KEYS.sort()

    if (DF_OBJECT_KEYS == S3_OBJECT_KEYS).all():
        print("ALL KEYS MATCHES")
    else:
        print("Keys from dataframe does not match keys on Bucket")
