
import torch.nn.functional as F
import torch
import timm


from models.networks import LandmarkResidual


def test_output_of_network() -> None:
    """Test output of our pretrained network is same as timm's output"""

    resnet50 = timm.create_model('resnet50', True)
    custom_resnet50 = LandmarkResidual(model='resnet50', pretrained=True, num_classes=1000)
    data = torch.randn(8, 3, 300, 300)
    features = resnet50.forward_features(data)
    custom_features = custom_resnet50.net.forward_features(data)
    pooled_features = F.adaptive_avg_pool2d(features, 1).view(8, -1)
    custom_pooled_features = custom_resnet50.pooling(custom_features).view(8, -1)
    assert (features == custom_features).all().item()
    assert (pooled_features == custom_pooled_features).all().item()
