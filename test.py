"""Central evaluation entry point for clean, corruption, PGD, and transfer testing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from attacks import pgd_attack
from features import PenultimateFeatureRecorder, build_feature_rows, save_feature_export
from models import build_model
from parameters import ExperimentConfig
from robustness import CORRUPTIONS, get_cifar10c_loader
from train import (
    append_run_row,
    build_log_row,
    get_loaders,
    get_teacher_run_name,
    upsert_summary_row,
)


EVAL_COLUMNS = [
    "run_name",
    "model_name",
    "source_model",
    "target_model",
    "checkpoint_path",
    "eval_type",
    "corruption_name",
    "severity",
    "attack_type",
    "attack_norm",
    "epsilon",
    "steps",
    "num_samples",
    "loss",
    "clean_accuracy",
    "adversarial_accuracy",
    "feature_path",
    "sample_path",
]


class NormalizedModelWrapper(nn.Module):
    """Wrap a model that expects normalized inputs with pixel-space normalization."""

    def __init__(self, model: nn.Module, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.mean) / self.std
        return self.model(normalized)


def _load_state_dict(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def _normalize_batch(images: torch.Tensor, params: ExperimentConfig) -> torch.Tensor:
    mean = torch.tensor(params["mean"], device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(params["std"], device=images.device).view(1, -1, 1, 1)
    return (images - mean) / std


def _denormalize_batch(images: torch.Tensor, params: ExperimentConfig) -> torch.Tensor:
    mean = torch.tensor(params["mean"], device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(params["std"], device=images.device).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def _resolve_checkpoint_path(params: ExperimentConfig) -> str:
    return params["checkpoint_path"] or params["student_checkpoint"] or params["save_path"]


def _resolve_target_checkpoint_path(params: ExperimentConfig) -> str:
    return params["target_checkpoint"] or params["student_checkpoint"] or params["checkpoint_path"] or params["save_path"]


def _resolve_source_checkpoint_path(params: ExperimentConfig) -> str:
    return params["source_checkpoint"] or params["teacher_checkpoint"]


def _load_existing_summary_row(params: ExperimentConfig) -> dict[str, str]:
    summary_path = Path(params["summary_path"])
    if not summary_path.exists():
        return {}
    with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("run_name") == params["run_name"]:
                return row
    return {}


def _evaluate_loader(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, int]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.detach().item() * images.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total, total


def _truncate_feature_row(row: dict[str, Any], limit: int) -> dict[str, Any]:
    return {key: value[:limit] for key, value in row.items()}


def _build_eval_path(params: ExperimentConfig, suffix: str) -> Path:
    return Path(params["robustness_dir"]) / f"{params['run_name']}_{suffix}.csv"


def _write_eval_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_gradcam_samples(path: Path, samples: list[dict[str, Any]]) -> Path | None:
    if not samples:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)
    return path


def run_clean_evaluation(
    model: nn.Module,
    params: ExperimentConfig,
    device: torch.device,
    train_summary: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate one checkpoint on the clean held-out test set."""

    _, _, test_loader = get_loaders(params)
    _load_state_dict(model, _resolve_checkpoint_path(params), device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, total = _evaluate_loader(model, test_loader, criterion, device)

    append_run_row(
        params,
        build_log_row(
            params=params,
            epoch=params["epochs"],
            split="test",
            loss=test_loss,
            accuracy=test_accuracy,
            lr=0.0,
        ),
    )

    existing_summary = _load_existing_summary_row(params)
    summary = existing_summary | (train_summary or {})
    summary.update(
        {
            "run_name": params["run_name"],
            "model": params["model"],
            "dataset": params["dataset"],
            "train_mode": params["train_mode"],
            "pretrained": params["pretrained"],
            "transfer_mode": params["transfer_mode"],
            "teacher_run": get_teacher_run_name(params),
            "checkpoint_path": _resolve_checkpoint_path(params),
            "augmix_enabled": params["augmix_enabled"],
            "label_smoothing": params["label_smoothing"],
            "temperature": params["temperature"] if params["train_mode"] == "distill" else existing_summary.get("temperature", ""),
            "alpha": params["alpha"] if params["train_mode"] == "distill" else existing_summary.get("alpha", ""),
            "test_loss": f"{test_loss:.6f}",
            "test_accuracy": f"{test_accuracy:.6f}",
            "seed": params["seed"],
        }
    )
    upsert_summary_row(params, summary)

    eval_row = {
        "run_name": params["run_name"],
        "model_name": params["model"],
        "source_model": "",
        "target_model": params["model"],
        "checkpoint_path": _resolve_checkpoint_path(params),
        "eval_type": "clean",
        "corruption_name": "clean",
        "severity": 0,
        "attack_type": "",
        "attack_norm": "",
        "epsilon": "",
        "steps": "",
        "num_samples": total,
        "loss": f"{test_loss:.6f}",
        "clean_accuracy": f"{test_accuracy:.6f}",
        "adversarial_accuracy": "",
        "feature_path": "",
        "sample_path": "",
    }
    _write_eval_rows(_build_eval_path(params, "clean"), [eval_row])

    print(f"Clean test | loss={test_loss:.4f} acc={test_accuracy:.4f}")
    return {"test_loss": test_loss, "test_accuracy": test_accuracy}


def run_cifar10c_evaluation(model: nn.Module, params: ExperimentConfig, device: torch.device) -> Path:
    """Evaluate one checkpoint on clean CIFAR-10 and CIFAR-10-C."""

    if params["dataset"] != "cifar10":
        raise ValueError("CIFAR-10-C evaluation is only supported for dataset=cifar10.")

    _load_state_dict(model, _resolve_checkpoint_path(params), device)
    _, _, clean_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()

    rows: list[dict[str, Any]] = []
    clean_loss, clean_acc, clean_total = _evaluate_loader(model, clean_loader, criterion, device)
    rows.append(
        {
            "run_name": params["run_name"],
            "model_name": params["model"],
            "source_model": "",
            "target_model": params["model"],
            "checkpoint_path": _resolve_checkpoint_path(params),
            "eval_type": "cifar10c",
            "corruption_name": "clean",
            "severity": 0,
            "attack_type": "",
            "attack_norm": "",
            "epsilon": "",
            "steps": "",
            "num_samples": clean_total,
            "loss": f"{clean_loss:.6f}",
            "clean_accuracy": f"{clean_acc:.6f}",
            "adversarial_accuracy": "",
            "feature_path": "",
            "sample_path": "",
        }
    )

    for corruption in CORRUPTIONS:
        for severity in params["cifar10c_severities"]:
            loader = get_cifar10c_loader(params, corruption=corruption, severity=severity)
            loss, accuracy, total = _evaluate_loader(model, loader, criterion, device)
            rows.append(
                {
                    "run_name": params["run_name"],
                    "model_name": params["model"],
                    "source_model": "",
                    "target_model": params["model"],
                    "checkpoint_path": _resolve_checkpoint_path(params),
                    "eval_type": "cifar10c",
                    "corruption_name": corruption,
                    "severity": severity,
                    "attack_type": "",
                    "attack_norm": "",
                    "epsilon": "",
                    "steps": "",
                    "num_samples": total,
                    "loss": f"{loss:.6f}",
                    "clean_accuracy": f"{accuracy:.6f}",
                    "adversarial_accuracy": "",
                    "feature_path": "",
                    "sample_path": "",
                }
            )

    output_path = _write_eval_rows(_build_eval_path(params, "cifar10c"), rows)
    print(f"Saved CIFAR-10-C evaluation to {output_path}")
    return output_path


def _run_pgd_like_evaluation(
    source_model: nn.Module,
    target_model: nn.Module,
    params: ExperimentConfig,
    device: torch.device,
    eval_type: str,
    target_checkpoint_path: str,
) -> Path:
    """Shared evaluation path for white-box PGD and teacher-to-student transfer."""

    _, _, test_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    source_wrapper = NormalizedModelWrapper(source_model, params["mean"], params["std"]).to(device)
    source_wrapper.eval()
    target_model.eval()

    feature_rows: list[dict[str, Any]] = []
    feature_budget = params["feature_export_limit"] if params["feature_export_limit"] > 0 else None
    recorded_features = 0
    feature_path: Path | None = None
    sample_path: Path | None = None
    samples: list[dict[str, Any]] = []

    feature_recorder = PenultimateFeatureRecorder(target_model) if params["export_features"] else None

    total_loss, total_clean_correct, total_adv_correct, total_samples = 0.0, 0, 0, 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pixel_images = _denormalize_batch(images, params)

        with torch.no_grad():
            clean_logits = target_model(images)
        clean_predictions = clean_logits.argmax(1)
        total_clean_correct += clean_predictions.eq(labels).sum().item()

        if feature_recorder is not None:
            clean_features = feature_recorder.take()
        else:
            clean_features = None

        adv_pixel = pgd_attack(
            model=source_wrapper,
            images=pixel_images,
            labels=labels,
            epsilon=params["attack_epsilon"],
            steps=params["attack_steps"],
            step_size=params["attack_step_size"],
            norm=params["attack_norm"],
            random_start=params["attack_random_start"],
        )
        adv_images = _normalize_batch(adv_pixel, params)

        with torch.no_grad():
            adv_logits = target_model(adv_images)
        adv_predictions = adv_logits.argmax(1)
        adv_loss = criterion(adv_logits, labels)
        total_loss += adv_loss.detach().item() * images.size(0)
        total_adv_correct += adv_predictions.eq(labels).sum().item()
        total_samples += images.size(0)

        if feature_recorder is not None:
            adv_features = feature_recorder.take()
            remaining = None if feature_budget is None else max(feature_budget - recorded_features, 0)
            if remaining != 0:
                clean_row = build_feature_rows(
                    features=clean_features,
                    labels=labels,
                    predictions=clean_predictions,
                    sample_kind="clean",
                    model_name=params["target_model"] if eval_type == "transfer" else params["model"],
                    run_name=params["run_name"],
                    attack_norm=params["attack_norm"],
                )
                adv_row = build_feature_rows(
                    features=adv_features,
                    labels=labels,
                    predictions=adv_predictions,
                    sample_kind="adversarial",
                    model_name=params["target_model"] if eval_type == "transfer" else params["model"],
                    run_name=params["run_name"],
                    attack_norm=params["attack_norm"],
                )
                if remaining is not None:
                    pairs_to_keep = min(clean_row["labels"].shape[0], remaining)
                    clean_row = _truncate_feature_row(clean_row, pairs_to_keep)
                    adv_row = _truncate_feature_row(adv_row, pairs_to_keep)
                feature_rows.extend([clean_row, adv_row])
                recorded_features += clean_row["labels"].shape[0]

        if params["gradcam_sample_count"] > 0 and len(samples) < params["gradcam_sample_count"]:
            clean_correct_adv_wrong = clean_predictions.eq(labels) & adv_predictions.ne(labels)
            selected = clean_correct_adv_wrong.nonzero(as_tuple=False).flatten()
            for index in selected.tolist():
                if len(samples) >= params["gradcam_sample_count"]:
                    break
                samples.append(
                    {
                        "clean_image": images[index].detach().cpu(),
                        "adv_image": adv_images[index].detach().cpu(),
                        "true_label": int(labels[index].item()),
                        "clean_prediction": int(clean_predictions[index].item()),
                        "adv_prediction": int(adv_predictions[index].item()),
                        "model_name": params["target_model"] if eval_type == "transfer" else params["model"],
                        "run_name": params["run_name"],
                        "checkpoint_path": target_checkpoint_path,
                        "attack_norm": params["attack_norm"],
                        "attack_epsilon": params["attack_epsilon"],
                    }
                )

    if feature_recorder is not None:
        feature_recorder.close()

    if feature_rows:
        feature_path = save_feature_export(
            Path(params["features_dir"]) / f"{params['run_name']}_{eval_type}_{params['attack_norm']}.npz",
            feature_rows,
        )

    if samples:
        sample_path = _save_gradcam_samples(
            Path(params["robustness_dir"]) / f"{params['run_name']}_{eval_type}_{params['attack_norm']}_samples.pt",
            samples,
        )

    adv_loss = total_loss / total_samples
    clean_acc = total_clean_correct / total_samples
    adv_acc = total_adv_correct / total_samples

    row = {
        "run_name": params["run_name"],
        "model_name": params["target_model"] if eval_type == "transfer" else params["model"],
        "source_model": params["source_model"] if eval_type == "transfer" else params["model"],
        "target_model": params["target_model"] if eval_type == "transfer" else params["model"],
        "checkpoint_path": target_checkpoint_path,
        "eval_type": eval_type,
        "corruption_name": "",
        "severity": "",
        "attack_type": params["attack_type"],
        "attack_norm": params["attack_norm"],
        "epsilon": params["attack_epsilon"],
        "steps": params["attack_steps"],
        "num_samples": total_samples,
        "loss": f"{adv_loss:.6f}",
        "clean_accuracy": f"{clean_acc:.6f}",
        "adversarial_accuracy": f"{adv_acc:.6f}",
        "feature_path": str(feature_path) if feature_path is not None else "",
        "sample_path": str(sample_path) if sample_path is not None else "",
    }
    output_path = _write_eval_rows(
        _build_eval_path(params, f"{eval_type}_{params['attack_norm']}"),
        [row],
    )

    print(
        f"{eval_type.upper()} {params['attack_norm']} | "
        f"clean_acc={clean_acc:.4f} adv_acc={adv_acc:.4f}"
    )
    return output_path


def run_pgd_evaluation(model: nn.Module, params: ExperimentConfig, device: torch.device) -> Path:
    """Run white-box PGD evaluation for the currently selected checkpoint."""

    checkpoint_path = _resolve_checkpoint_path(params)
    _load_state_dict(model, checkpoint_path, device)
    return _run_pgd_like_evaluation(
        source_model=model,
        target_model=model,
        params=params,
        device=device,
        eval_type="pgd",
        target_checkpoint_path=checkpoint_path,
    )


def run_transfer_evaluation(target_model: nn.Module, params: ExperimentConfig, device: torch.device) -> Path:
    """Generate teacher-crafted adversarial examples and evaluate them on the student."""

    source_checkpoint = _resolve_source_checkpoint_path(params)
    if not source_checkpoint:
        raise ValueError("Transfer evaluation requires --source_checkpoint or --teacher_checkpoint.")

    target_checkpoint = _resolve_target_checkpoint_path(params)
    if not target_checkpoint:
        raise ValueError("Transfer evaluation requires a target checkpoint.")

    source_model = build_model(
        params,
        model_name=params["source_model"],
        pretrained=params["source_pretrained"],
        transfer_mode=params["source_transfer_mode"],
        freeze_backbone=params["source_freeze_backbone"],
    ).to(device)
    _load_state_dict(source_model, source_checkpoint, device)
    _load_state_dict(target_model, target_checkpoint, device)

    return _run_pgd_like_evaluation(
        source_model=source_model,
        target_model=target_model,
        params=params,
        device=device,
        eval_type="transfer",
        target_checkpoint_path=target_checkpoint,
    )


def run_test(
    model: torch.nn.Module,
    params: ExperimentConfig,
    device: torch.device,
    train_summary: dict[str, Any] | None = None,
):
    """Dispatch to the configured evaluation mode."""

    if params["eval_mode"] == "clean":
        return run_clean_evaluation(model, params, device, train_summary=train_summary)
    if params["eval_mode"] == "cifar10c":
        return run_cifar10c_evaluation(model, params, device)
    if params["eval_mode"] == "pgd":
        return run_pgd_evaluation(model, params, device)
    if params["eval_mode"] == "transfer":
        return run_transfer_evaluation(model, params, device)
    raise ValueError(f"Unsupported eval_mode: {params['eval_mode']}")
