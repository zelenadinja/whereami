
import torch.nn as nn
import torch


def criterion(
        class_weights: torch.Tensor = None,
        reduction: str = "mean",
) -> torch.Tensor:
    """CrossEntropyLoss"""

    crit = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
    return crit
