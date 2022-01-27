"""Testing loading weights into model"""

import pytest
import timm

from models.utils import load_weights_from_s3


@pytest.mark.parametrize("model_name", ["resnet26", "resnet34", "resnet50"])
@pytest.mark.parametrize("pretrained", [True, False])
def test_weights_load(model_name: str, pretrained: bool) -> None:
    """Test that weights from pretrained models matches"""
    weights = load_weights_from_s3(model_name=model_name)
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
