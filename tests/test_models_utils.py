# type: ignore
from typing import Optional

import pytest
import timm

from models.utils import load_weights_from_s3


@pytest.mark.parametrize("weights_object_key", ["pretrainedweights/resnet26", "pretrainedweights/resnet34", "pretrainedweights/resnet50"])
@pytest.mark.parametrize("pretrained", [True, False])
def test_weights_load(weights_object_key: str, pretrained: bool) -> None:
    """Test that weights from pretrained models matches"""
    weights: Optional[dict] = load_weights_from_s3(model_name=model_name)
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
