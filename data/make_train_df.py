"""Make a training dataframe"""

import os

import pandas as pd
from dotenv import load_dotenv

from data.utils import get_landmark_ids, get_image_fpaths, label_encoder
from src.utils import artifact_to_s3


if __name__ == '__main__':

    load_dotenv()
    DF = pd.read_csv(os.environ.get('TRAIN_CSV'))
    DF['object_key'] = get_image_fpaths(dataframe=DF, train=True)
    LANDMARKS = get_landmark_ids(dataframe=DF, num_images=200)
    USED_LANDMARKS = LANDMARKS['used_landmarks']
    NOT_USED_LANDMARKS = LANDMARKS['not_used_landmarks']
    USED_OBJECT_KEYS = DF[DF['landmark_id'].isin(USED_LANDMARKS)]['object_key'].tolist()
    NOT_USED_OBJECT_KEYS = DF[DF['landmark_id'].isin(NOT_USED_LANDMARKS)]['object_key'].tolist()
    DF = DF[DF['landmark_id'].isin(USED_LANDMARKS)].reset_index(drop=True)
    ENC = label_encoder(dataframe=DF, target_column='landmark_id')
    DF['target'] = ENC['encoded_target']
    LABEL_ENCODER = ENC['la']

    #Save artifacts
    artifact_to_s3(object_=LABEL_ENCODER, bucket=os.environ.get('S3_BUCKET'), key='label_encoder', extension='pkl')
    artifact_to_s3(object_=USED_LANDMARKS, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/used_landmarks')
    artifact_to_s3(object_=NOT_USED_LANDMARKS, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/not_used_landmarks')
    artifact_to_s3(object_=USED_OBJECT_KEYS, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/used_object_keys')
    artifact_to_s3(object_=NOT_USED_OBJECT_KEYS, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/not_used_object_keys')

    #Uplaod csv to S3
    DF.to_csv(os.environ.get('PROCESSED_TRAIN_CSV'), index=False)



