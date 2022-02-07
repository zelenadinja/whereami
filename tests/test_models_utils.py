import pytest
import timm

from models.utils import load_weights_from_s3


@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize(
    "model_name",
    ['resnet26', 'resnet34', 'resnet50'],
)
def test_model_weights(model_name, pretrained):
    """Test custom loading weights by comparing
    it with timm's weights
    """
    weights = load_weights_from_s3(
        weights_object_key=f'pretrainedweights/{model_name}.pth'
    )
    timm_model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained
    )
    timm_weights = timm_model.state_dict()

    assert len(weights) == len(timm_weights)

    if pretrained is True:
        assert (
            (weights["conv1.weight"] == timm_weights["conv1.weight"])
            .all()
            .item()
        )
    if pretrained is False:
        assert (
            weights["conv1.weight"] == timm_weights["conv1.weight"]
        ).all().item() is False
