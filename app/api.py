"""Flask web server serving landmark recognition predicionts"""

import os

import boto3
from flask import request
from flask import Flask
from flask import render_template
from dotenv import load_dotenv

from src.utils import read_image_s3, read_artifacts_s3
from app.app_prediction import get_prediction
from models.networks import LandmarkResidual
from models.utils import load_weights_from_s3

load_dotenv()
app = Flask(__name__)
TARGET2CATEGORY= read_artifacts_s3(object_key='target2category.json')

S3CLIENT = boto3.client('s3')
BUCKET_NAME = os.environ.get('S3_BUCKET_INPUTS')
MODEL = LandmarkResidual(
    model=os.environ.get('API_MODEL'),
    weights_object_key=None,
    num_classes=int(os.environ.get('NUM_CLASSES')),
)
WEIGHTS = load_weights_from_s3(
    weights_object_key=os.environ.get('API_WEIGHTS'),
)
MODEL.load_state_dict(WEIGHTS)


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            S3CLIENT.upload_fileobj(
                file,
                BUCKET_NAME,
                file.filename,
            )
            image = read_image_s3(
                object_key=file.filename,
                bucket_name=BUCKET_NAME,
            )
            prediction = get_prediction(model=MODEL, image=image)
            label = TARGET2CATEGORY[str(prediction)]
            return render_template(
                'index.html',
                prediction=label,
                image_loc=file.filename
            )
    return render_template('index.html', prediction=0, image_loc=None)


if __name__ == '__main__':

    app.run(port=12000, debug=False)
