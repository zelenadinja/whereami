# type: ignore
from typing import Optional

import pytest
import timm

from models.utils import load_weights_from_s3


@pytest.mark.parametrize(
    "weights_object_key",
    [
        "pretrainedweights/resnet26",
        "pretrainedweights/resnet34",
        "pretrainedweights/resnet50"]
    )
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("model_name", ['resnet26', 'resnet34', 'resnet50'])
def test_weights_load(
    weights_object_key: str, pretrained: bool, model_name: str
) -> None:
    """Test that weights from pretrained models matches"""
    weights: Optional[dict] = load_weights_from_s3(
        weights_object_key=weights_object_key
    )
    resnet = timm.create_model(model_name=model_name, pretrained=pretrained)

    assert len(weights) == len(resnet.state_dict())
    if pretrained is True:
        assert (
            (weights["conv1.weight"] == resnet.state_dict()["conv1.weight"])
            .all()
            .item()
        )
    if pretrained is False:
        assert (
            weights["conv1.weight"] == resnet.state_dict()["conv1.weight"]
        ).all().item() is False
