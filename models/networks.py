from typing import Optional

import timm
import torch
import torch.nn as nn

from models.utils import load_weights_from_s3


class LandmarkResidual(nn.Module):

    def __init__(
        self, model: str, weights_object_key: Optional[str], num_classes: int
    ) -> None:
        super().__init__()

        self.net = timm.create_model(model, pretrained=False)
        if weights_object_key:
            weights = load_weights_from_s3(
                weights_object_key=weights_object_key
            )
            self.net.load_state_dict(weights)

        n_features = self.net.fc.in_features
        self.net.global_pool = nn.Identity()
        self.net.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through the data"""
        batch_size = x.size(0)
        features = self.net.forward_features(x)
        pooled_features = self.pooling(features).view(batch_size, -1)
        output = self.last_linear(pooled_features)

        return output
