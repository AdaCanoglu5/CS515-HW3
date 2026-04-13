"""CIFAR-10-C loading helpers and shared evaluation metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


class CIFAR10CDataset(Dataset):
    """Dataset for one CIFAR-10-C corruption and severity."""

    def __init__(self, root: str, corruption: str, severity: int, transform=None) -> None:
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        data_path = self.root / f"{corruption}.npy"
        labels_path = self.root / "labels.npy"
        if not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "CIFAR-10-C files were not found. Expected files like "
                f"{data_path} and {labels_path}."
            )

        all_images = np.load(data_path)
        all_labels = np.load(labels_path)

        start = (severity - 1) * 10000
        end = severity * 10000
        self.images = all_images[start:end]
        self.labels = all_labels[start:end]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_cifar10c_loader(params, corruption: str, severity: int) -> DataLoader:
    """Build a deterministic evaluation loader for one corruption slice."""

    from train import get_transforms

    dataset = CIFAR10CDataset(
        root=params["cifar10c_dir"],
        corruption=corruption,
        severity=severity,
        transform=get_transforms(params, train=False),
    )
    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=False,
    )
