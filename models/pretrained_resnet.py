"""Kept from HW2 for compatibility with the previous project layout."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_pretrained_resnet18(num_classes: int) -> nn.Module:
    """Build a torchvision ResNet-18 with a CIFAR-10-sized classifier head."""

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
