
import torch
import torch.nn as nn


def criterion(
        class_weights: torch.Tensor = None,
        reduction: str = "mean",
) -> torch.Tensor:
    """CrossEntropyLoss"""

    crit = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
    return crit
