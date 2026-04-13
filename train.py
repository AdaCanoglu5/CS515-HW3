"""Training utilities for clean, AugMix, and distillation experiments."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from augmix import AugMixDataset
from models import build_model
from parameters import ExperimentConfig


RUN_CSV_COLUMNS = [
    "run_name",
    "epoch",
    "split",
    "loss",
    "accuracy",
    "lr",
    "seed",
    "model",
    "dataset",
    "train_mode",
    "pretrained",
    "transfer_mode",
    "teacher_run",
    "augmix_enabled",
    "label_smoothing",
    "temperature",
    "alpha",
]

SUMMARY_COLUMNS = [
    "run_name",
    "model",
    "dataset",
    "train_mode",
    "pretrained",
    "transfer_mode",
    "teacher_run",
    "checkpoint_path",
    "augmix_enabled",
    "label_smoothing",
    "temperature",
    "alpha",
    "best_epoch",
    "best_val_loss",
    "best_val_accuracy",
    "final_train_loss",
    "final_train_accuracy",
    "final_val_loss",
    "final_val_accuracy",
    "test_loss",
    "test_accuracy",
    "seed",
]


def ensure_output_dirs(params: ExperimentConfig) -> None:
    """Create the required artifact directories."""

    for path in (
        params["output_dir"],
        params["checkpoint_dir"],
        params["run_dir"],
        params["robustness_dir"],
        params["features_dir"],
        params["figures_dir"],
        params["config_dir"],
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def write_config_snapshot(params: ExperimentConfig) -> None:
    """Persist the active config as JSON for reproducibility."""

    if not params["save_config"]:
        return
    config_path = Path(params["config_path"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as output_file:
        json.dump(params.to_dict(), output_file, indent=2)


def get_teacher_run_name(params: ExperimentConfig) -> str:
    """Infer the teacher identifier from the configured checkpoint."""

    teacher_checkpoint = params["teacher_checkpoint"]
    return Path(teacher_checkpoint).stem if teacher_checkpoint else ""


def _resolve_mean_std(params: ExperimentConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Match HW2 normalization logic for transfer-learning runs."""

    if params["transfer_mode"] == "resize_freeze":
        return params["imagenet_mean"], params["imagenet_std"]
    return params["mean"], params["std"]


def get_transforms(params: ExperimentConfig, train: bool = True) -> transforms.Compose:
    """Build dataset transforms, preserving the HW2 transfer branches."""

    mean, std = _resolve_mean_std(params)

    if params["dataset"] == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    if params["transfer_mode"] == "resize_freeze":
        steps: list[Any] = [transforms.Resize((params["input_size"], params["input_size"]))]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transforms.Compose(steps)

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(params["input_size"], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _get_cifar_pil_augment(params: ExperimentConfig) -> transforms.Compose:
    if params["transfer_mode"] == "resize_freeze":
        return transforms.Compose(
            [
                transforms.Resize((params["input_size"], params["input_size"])),
                transforms.RandomHorizontalFlip(),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomCrop(params["input_size"], padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )


def _split_indices(total_items: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    val_size = int(total_items * val_split)
    if val_size <= 0 or val_size >= total_items:
        raise ValueError("val_split must leave at least one sample in both train and validation splits.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_items, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def _build_datasets(params: ExperimentConfig):
    """Create reproducible train, validation, and test datasets."""

    if params["dataset"] == "mnist":
        train_tf = get_transforms(params, train=True)
        eval_tf = get_transforms(params, train=False)
        raw_train = datasets.MNIST(params["data_dir"], train=True, download=True, transform=train_tf)
        raw_val = datasets.MNIST(params["data_dir"], train=True, download=True, transform=eval_tf)
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=eval_tf)
        train_indices, val_indices = _split_indices(len(raw_train), params["val_split"], params["seed"])
        return Subset(raw_train, train_indices), Subset(raw_val, val_indices), test_ds

    eval_tf = get_transforms(params, train=False)
    train_indices, val_indices = _split_indices(50000, params["val_split"], params["seed"])

    if params["train_mode"] == "augmix_finetune" or params["augmix_enabled"]:
        base_train = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=None)
        augmix_train = AugMixDataset(
            base_dataset=base_train,
            train_transform=_get_cifar_pil_augment(params),
            preprocess=eval_tf,
            severity=params["augmix_severity"],
            width=params["augmix_width"],
            depth=params["augmix_depth"],
            alpha=params["augmix_alpha"],
        )
        train_ds = Subset(augmix_train, train_indices)
    else:
        clean_train = datasets.CIFAR10(
            params["data_dir"],
            train=True,
            download=True,
            transform=get_transforms(params, train=True),
        )
        train_ds = Subset(clean_train, train_indices)

    val_base = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=eval_tf)
    test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=eval_tf)
    val_ds = Subset(val_base, val_indices)
    return train_ds, val_ds, test_ds


def get_loaders(params: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test loaders."""

    train_ds, val_ds, test_ds = _build_datasets(params)
    loader_kwargs = {
        "batch_size": params["batch_size"],
        "num_workers": params["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def reset_run_csv(params: ExperimentConfig) -> None:
    """Create or overwrite the per-run metrics CSV."""

    csv_path = Path(params["csv_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_CSV_COLUMNS)
        writer.writeheader()


def append_run_row(params: ExperimentConfig, row: dict[str, Any]) -> None:
    """Append one row to the per-run CSV."""

    if not params["save_csv"]:
        return
    with Path(params["csv_path"]).open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_CSV_COLUMNS)
        writer.writerow(row)


def upsert_summary_row(params: ExperimentConfig, row: dict[str, Any]) -> None:
    """Replace or append one row in the compact summary CSV."""

    summary_path = Path(params["summary_path"])
    existing_rows: list[dict[str, Any]] = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))

    filtered_rows = [existing for existing in existing_rows if existing.get("run_name") != params["run_name"]]
    filtered_rows.append({column: row.get(column, "") for column in SUMMARY_COLUMNS})

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(filtered_rows)


def build_log_row(
    params: ExperimentConfig,
    epoch: int,
    split: str,
    loss: float,
    accuracy: float,
    lr: float,
) -> dict[str, Any]:
    """Build one CSV metrics row."""

    teacher_run = get_teacher_run_name(params)
    temperature = params["temperature"] if params["train_mode"] == "distill" else ""
    alpha = params["alpha"] if params["train_mode"] == "distill" else ""
    return {
        "run_name": params["run_name"],
        "epoch": epoch,
        "split": split,
        "loss": f"{loss:.6f}",
        "accuracy": f"{accuracy:.6f}",
        "lr": f"{lr:.8f}",
        "seed": params["seed"],
        "model": params["model"],
        "dataset": params["dataset"],
        "train_mode": params["train_mode"],
        "pretrained": params["pretrained"],
        "transfer_mode": params["transfer_mode"],
        "teacher_run": teacher_run,
        "augmix_enabled": params["augmix_enabled"],
        "label_smoothing": params["label_smoothing"],
        "temperature": temperature,
        "alpha": alpha,
    }


def load_teacher_model(params: ExperimentConfig, device: torch.device) -> nn.Module:
    """Load the teacher checkpoint used for distillation."""

    teacher_checkpoint = params["teacher_checkpoint"]
    if not teacher_checkpoint:
        raise ValueError("A teacher checkpoint is required when train_mode=distill.")

    teacher = build_model(
        params,
        model_name=params["teacher_model"],
        pretrained=params["teacher_pretrained"],
        transfer_mode=params["teacher_transfer_mode"],
        freeze_backbone=params["teacher_freeze_backbone"],
    ).to(device)
    checkpoint = torch.load(teacher_checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute standard knowledge-distillation loss."""

    hard_loss = criterion(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    return (1.0 - alpha) * hard_loss + alpha * (temperature**2) * soft_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    params: ExperimentConfig,
    teacher_model: nn.Module | None = None,
) -> tuple[float, float]:
    """Run one training epoch."""

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        student_logits = model(images)

        if params["train_mode"] == "distill":
            if teacher_model is None:
                raise ValueError("teacher_model is required for distillation.")
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            loss = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=params["temperature"],
                alpha=params["alpha"],
                criterion=criterion,
            )
        else:
            loss = criterion(student_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)
        correct += student_logits.argmax(1).eq(labels).sum().item()
        total += images.size(0)

        if params["log_interval"] and (batch_idx + 1) % params["log_interval"] == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] loss={total_loss / total:.4f} acc={correct / total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation or test loader."""

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.detach().item() * images.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def _save_checkpoint(
    params: ExperimentConfig,
    state_dict: dict[str, torch.Tensor],
    epoch: int,
    best_val_accuracy: float,
) -> None:
    checkpoint = {
        "model_state_dict": state_dict,
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
        "config_path": params["config_path"],
        "run_name": params["run_name"],
    }
    torch.save(checkpoint, params["save_path"])


def run_training(
    model: nn.Module,
    params: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run the main training loop and save the best validation checkpoint."""

    ensure_output_dirs(params)
    write_config_snapshot(params)
    train_loader, val_loader, _ = get_loaders(params)
    if params["save_csv"]:
        reset_run_csv(params)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss()
    teacher_model = load_teacher_model(params, device) if params["train_mode"] == "distill" else None

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = -1.0
    best_epoch = 0
    best_val_loss = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    final_train_loss = 0.0
    final_train_acc = 0.0
    final_val_loss = 0.0
    final_val_acc = 0.0

    for epoch in range(1, params["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            device=device,
            params=params,
            teacher_model=teacher_model,
        )
        val_loss, val_acc = evaluate(model, val_loader, eval_criterion, device)
        scheduler.step()

        append_run_row(params, build_log_row(params, epoch, "train", train_loss, train_acc, current_lr))
        append_run_row(params, build_log_row(params, epoch, "val", val_loss, val_acc, current_lr))

        print(
            f"Epoch {epoch:02d}/{params['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        final_train_loss = train_loss
        final_train_acc = train_acc
        final_val_loss = val_loss
        final_val_acc = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            _save_checkpoint(params, best_weights, epoch, best_acc)

    model.load_state_dict(best_weights)
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

    summary = {
        "run_name": params["run_name"],
        "model": params["model"],
        "dataset": params["dataset"],
        "train_mode": params["train_mode"],
        "pretrained": params["pretrained"],
        "transfer_mode": params["transfer_mode"],
        "teacher_run": get_teacher_run_name(params),
        "checkpoint_path": params["save_path"],
        "augmix_enabled": params["augmix_enabled"],
        "label_smoothing": params["label_smoothing"],
        "temperature": params["temperature"] if params["train_mode"] == "distill" else "",
        "alpha": params["alpha"] if params["train_mode"] == "distill" else "",
        "best_epoch": best_epoch,
        "best_val_loss": f"{best_val_loss:.6f}",
        "best_val_accuracy": f"{best_acc:.6f}",
        "final_train_loss": f"{final_train_loss:.6f}",
        "final_train_accuracy": f"{final_train_acc:.6f}",
        "final_val_loss": f"{final_val_loss:.6f}",
        "final_val_accuracy": f"{final_val_acc:.6f}",
        "test_loss": "",
        "test_accuracy": "",
        "seed": params["seed"],
    }
    upsert_summary_row(params, summary)
    return summary
