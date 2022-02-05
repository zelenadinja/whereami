import os

from dotenv import load_dotenv
import numpy as np
import torch

from src.utils import read_artifacts_s3
from dataset.augmentations import aug_version_1

load_dotenv()
args = read_artifacts_s3(object_key=os.environ.get('VERSION_1'))


def get_prediction(model, image):
    """Generate prediction for uplaoded image"""
    model.eval()
    #evaluate mode
    
    augment = aug_version_1(config=args, train=True)
    augment_image = augment(image=image)['image']
    augment_image.unsqueeze_(0)
    predictions = model(augment_image)
    _, label = torch.max(predictions, dim=1)

    return label.item()
