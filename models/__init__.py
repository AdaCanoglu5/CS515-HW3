"""Model factory helpers for the HW3 experiment pipeline."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

from .CNN import MNIST_CNN, SimpleCNN
from .MLP import MLP
from .pretrained_resnet import build_pretrained_resnet18
from .ResNet import BasicBlock, ResNet
from .VGG import VGG
from .mobilenet import MobileNetV2


def build_model(
    params: Mapping[str, object],
    model_name: str | None = None,
    pretrained: bool | None = None,
    transfer_mode: str | None = None,
    freeze_backbone: bool | None = None,
) -> nn.Module:
    """Build a model from the shared config."""

    name = model_name or str(params["model"])
    dataset = str(params["dataset"])
    num_classes = int(params["num_classes"])
    use_pretrained = bool(params["pretrained"]) if pretrained is None else pretrained
    resolved_transfer_mode = str(params["transfer_mode"]) if transfer_mode is None else transfer_mode
    resolved_freeze_backbone = bool(params["freeze_backbone"]) if freeze_backbone is None else freeze_backbone

    if use_pretrained:
        if dataset != "cifar10" or name != "resnet":
            raise ValueError("Pretrained transfer learning is only supported for CIFAR-10 ResNet runs.")
        return build_pretrained_resnet18(
            num_classes=num_classes,
            transfer_mode=resolved_transfer_mode,
            freeze_backbone=resolved_freeze_backbone,
        )

    if name == "mlp":
        return MLP(
            input_size=int(params["feature_size"]),
            hidden_sizes=list(params["hidden_sizes"]),
            num_classes=num_classes,
            dropout=float(params["dropout"]),
        )

    if name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=num_classes)
        return SimpleCNN(num_classes=num_classes)

    if name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is only supported for 3-channel inputs.")
        return VGG(dept=str(params["vgg_depth"]), num_class=num_classes)

    if name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is only supported for 3-channel inputs.")
        return ResNet(BasicBlock, list(params["resnet_layers"]), num_classes=num_classes)

    if name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is only supported for 3-channel inputs.")
        return MobileNetV2(num_classes=num_classes)

    raise ValueError(f"Unknown model: {name}")


def resolve_classifier_module(model: nn.Module) -> nn.Module:
    """Return the last linear layer, used for penultimate feature hooks."""

    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("Could not find a classifier linear layer for the requested model.")
    return linear_layers[-1]


def resolve_last_conv_module(model: nn.Module) -> nn.Module:
    """Return the final convolutional layer for Grad-CAM."""

    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Could not find a convolutional layer for the requested model.")
    return conv_layers[-1]
