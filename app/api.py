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

S3CLIENT = boto3.client('s3')

TARGET2CATEGORY= read_artifacts_s3(object_key='target2category.json')
API_MODEL_NAME = os.environ['API_MODEL']
API_WEIGHTS_KEY = os.environ['API_WEIGHTS']
BUCKET_NAME = os.environ['S3_BUCKET_INPUTS']
NUM_CLASSES = int(os.environ['NUM_CLASSES'])


MODEL = LandmarkResidual(
    model=API_MODEL_NAME,
    weights_object_key=None,
    num_classes=NUM_CLASSES,
)

WEIGHTS = load_weights_from_s3(
    weights_object_key=API_WEIGHTS_KEY,
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
            confidence, predictions = get_prediction(model=MODEL, image=image)

            label1 = TARGET2CATEGORY[str(predictions[0].item())]
            label2 = TARGET2CATEGORY[str(predictions[1].item())]
            label3 = TARGET2CATEGORY[str(predictions[2].item())]
            conf1 = round(confidence[0].item(), 2)
            conf2 = round(confidence[1].item(), 2)
            conf3 = round(confidence[2].item(), 2)

            labels = (label1, label2, label3)
            confs = (conf1, conf2, conf3)

            return render_template(
                'index.html',
                labels=labels,
                confidences = confs,
                image_loc=file.filename
            )
    return render_template('index.html', prediction=0, image_loc=None)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
