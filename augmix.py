"""Minimal AugMix utilities for CIFAR-10 teacher training."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


def _int_parameter(level: int, max_value: int) -> int:
    return int(level * max_value / 10)


def _float_parameter(level: int, max_value: float) -> float:
    return float(level) * max_value / 10.0


def _sample_level(severity: int) -> float:
    return np.random.uniform(low=0.1, high=float(severity))


def autocontrast(image: Image.Image, _: int) -> Image.Image:
    return ImageOps.autocontrast(image)


def equalize(image: Image.Image, _: int) -> Image.Image:
    return ImageOps.equalize(image)


def posterize(image: Image.Image, severity: int) -> Image.Image:
    level = max(1, 4 - _int_parameter(int(_sample_level(severity)), 4))
    return ImageOps.posterize(image, level)


def rotate(image: Image.Image, severity: int) -> Image.Image:
    degrees = _int_parameter(int(_sample_level(severity)), 30)
    if random.random() > 0.5:
        degrees = -degrees
    return image.rotate(degrees)


def solarize(image: Image.Image, severity: int) -> Image.Image:
    level = 256 - _int_parameter(int(_sample_level(severity)), 192)
    return ImageOps.solarize(image, level)


def shear_x(image: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 0.3)
    if random.random() > 0.5:
        level = -level
    return image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def shear_y(image: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 0.3)
    if random.random() > 0.5:
        level = -level
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def translate_x(image: Image.Image, severity: int) -> Image.Image:
    level = _int_parameter(int(_sample_level(severity)), image.size[0] // 3)
    if random.random() > 0.5:
        level = -level
    return image.transform(image.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def translate_y(image: Image.Image, severity: int) -> Image.Image:
    level = _int_parameter(int(_sample_level(severity)), image.size[1] // 3)
    if random.random() > 0.5:
        level = -level
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def color(image: Image.Image, severity: int) -> Image.Image:
    factor = 1.0 + _float_parameter(_sample_level(severity), 0.9)
    if random.random() > 0.5:
        factor = 1.0 - (factor - 1.0)
    return ImageEnhance.Color(image).enhance(factor)


def contrast(image: Image.Image, severity: int) -> Image.Image:
    factor = 1.0 + _float_parameter(_sample_level(severity), 0.9)
    if random.random() > 0.5:
        factor = 1.0 - (factor - 1.0)
    return ImageEnhance.Contrast(image).enhance(factor)


def brightness(image: Image.Image, severity: int) -> Image.Image:
    factor = 1.0 + _float_parameter(_sample_level(severity), 0.9)
    if random.random() > 0.5:
        factor = 1.0 - (factor - 1.0)
    return ImageEnhance.Brightness(image).enhance(factor)


def sharpness(image: Image.Image, severity: int) -> Image.Image:
    factor = 1.0 + _float_parameter(_sample_level(severity), 0.9)
    if random.random() > 0.5:
        factor = 1.0 - (factor - 1.0)
    return ImageEnhance.Sharpness(image).enhance(factor)


AUGMIX_OPS: tuple[Callable[[Image.Image, int], Image.Image], ...] = (
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
)


def augment_and_mix(
    image: Image.Image,
    preprocess: Callable[[Image.Image], torch.Tensor],
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply AugMix to one PIL image and return a normalized tensor."""

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(preprocess(image))
    for branch_weight in ws:
        image_aug = image.copy()
        chain_depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(chain_depth):
            op = random.choice(AUGMIX_OPS)
            image_aug = op(image_aug, severity)
        mix += float(branch_weight) * preprocess(image_aug)

    clean = preprocess(image)
    return (1.0 - float(m)) * clean + float(m) * mix


class AugMixDataset(Dataset):
    """Dataset wrapper that applies standard CIFAR training augmentations plus AugMix."""

    def __init__(
        self,
        base_dataset: Dataset,
        train_transform: Callable[[Image.Image], Image.Image],
        preprocess: Callable[[Image.Image], torch.Tensor],
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
    ) -> None:
        self.base_dataset = base_dataset
        self.train_transform = train_transform
        self.preprocess = preprocess
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = self.train_transform(image)
        mixed = augment_and_mix(
            image=image,
            preprocess=self.preprocess,
            severity=self.severity,
            width=self.width,
            depth=self.depth,
            alpha=self.alpha,
        )
        return mixed, label
