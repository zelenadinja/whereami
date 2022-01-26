import pandas as pd 
import os
from dotenv import load_dotenv
from data.utils import get_landmark_ids, get_image_fpaths, label_encoder
from src.utils import artifact_to_s3


if __name__ == '__main__':

    load_dotenv()
    df = pd.read_csv(os.environ.get('TRAIN_CSV'))                 
    df['object_key'] = get_image_fpaths(dataframe=df, train=True) 
    landmarks = get_landmark_ids(dataframe=df, num_images=200)    
    used_landmarks = landmarks['used_landmarks'] 
    not_used_landmarks = landmarks['not_used_landmarks']
    used_object_keys = df[df['landmark_id'].isin(used_landmarks)]['object_key'].tolist()       
    not_used_object_keys = df[df['landmark_id'].isin(not_used_landmarks)]['object_key'].tolist()
    df = df[df['landmark_id'].isin(used_landmarks)].reset_index(drop=True)
    enc = label_encoder(dataframe=df, target_column='landmark_id')
    df['target'] = enc['encoded_target']
    label_encoder = enc['la']

    #Save artifacts
    artifact_to_s3(object_=label_encoder, bucket=os.environ.get('S3_BUCKET'), key='label_encoder', extension='pkl')
    artifact_to_s3(object_=used_landmarks, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/used_landmarks')
    artifact_to_s3(object_=not_used_landmarks, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/not_used_landmarks')
    artifact_to_s3(object_=used_object_keys, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/used_object_keys')
    artifact_to_s3(object_=not_used_object_keys, bucket=os.environ.get('S3_BUCKET'), key='df_artifacts/not_used_object_keys')

    #Uplaod csv to S3
    df.to_csv(os.environ.get('PROCESSED_TRAIN_CSV'), index=False) 



