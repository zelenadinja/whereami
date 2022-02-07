import os

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from dataset.augmentations import aug_version_1
from src.utils import read_artifacts_s3

load_dotenv()
ARGS_PATH = os.environ.get('VERSION_1')

if ARGS_PATH:
    args = read_artifacts_s3(ARGS_PATH)


def get_prediction(model, image):
    """Generate prediction for uplaoded image"""

    model.eval()
    # evaluate mode
    augment = aug_version_1(config=args, train=True)
    augment_image = augment(image=image)['image']
    augment_image.unsqueeze_(0)

    with torch.no_grad():
        predictions = model(augment_image)
        soft_preds = F.softmax(predictions, dim=1)
        probs, labels = torch.topk(soft_preds, dim=1, k=3)

    return probs[0], labels[0]
