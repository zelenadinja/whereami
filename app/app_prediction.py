import os

from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn.functional as F

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

    with torch.no_grad():
        predictions = model(augment_image)
        soft_preds = F.softmax(predictions, dim=1)
        probs, labels = torch.topk(soft_preds, dim=1, k=3)

    return probs[0], labels[0]
