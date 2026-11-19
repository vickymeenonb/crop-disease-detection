"""Model construction with torchvision backbones."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    if name != "resnet50":
        raise ValueError(f"Unsupported model: {name}")

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
