import os

import pandas as pd
from dotenv import load_dotenv

from src.utils import artifact_to_s3

if __name__ == '__main__':

    load_dotenv()
    S3_BUCKET = os.environ['S3_BUCKET']
    
    categories = pd.read_csv(os.environ.get('LANDMARK2CAT'))
    categories['category_'] = categories['category'].apply(
        lambda cat: cat.split('Category:')[1]
    )
    df = pd.read_csv(os.environ.get('PROCESSED_TRAIN_CSV'))
    df['category'] = df['landmark_id'].map(categories['category_'])

    category2target = {
        cat: int(tar) for cat, tar in zip(
            df.category.unique(), df.target.unique()
        )
    }
    category2landmark = {
        cat: int(land) for cat, land in zip(
            df.category.unique(), df.landmark_id.unique()
        )
    }

    artifact_to_s3(
        object_=category2target, bucket=S3_BUCKET,
        key='category2target', extension='json', verbose=True
    )

    artifact_to_s3(
        object_=category2landmark, bucket=S3_BUCKET,
        key='category2landmark', extension='json', verbose=True
    )
