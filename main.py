"""Main entry point for training, evaluation, and FLOPs reporting."""

from __future__ import annotations

import csv
import random
import ssl
from pathlib import Path

import numpy as np
import torch

from models import build_model
from models.CNN import SimpleCNN
from models.ResNet import BasicBlock, ResNet
from parameters import ExperimentConfig, get_params
from test import run_test
from train import ensure_output_dirs, run_training, write_config_snapshot


ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Seed major random number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_flops_summary(params: ExperimentConfig, device: torch.device) -> Path:
    """Compute and save one FLOPs summary CSV for the teacher/student family."""

    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise ImportError("ptflops is required for --mode flops. Please install it first.") from exc

    output_path = Path(params["flops_summary_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    models_to_profile = [
        ("ResNetTeacher", ResNet(BasicBlock, params["resnet_layers"], num_classes=params["num_classes"]), (3, 32, 32)),
        ("SimpleCNNStudent", SimpleCNN(num_classes=params["num_classes"]), (3, 32, 32)),
    ]

    for arch_name, model, input_res in models_to_profile:
        model = model.to(device)
        macs, params_count = get_model_complexity_info(
            model,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        rows.append(
            {
                "arch_name": arch_name,
                "input_res": "x".join(str(dim) for dim in input_res),
                "flops": int(macs * 2),
                "macs": int(macs),
                "params": int(params_count),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["arch_name", "input_res", "flops", "macs", "params"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved FLOPs summary to {output_path}")
    return output_path


def _get_runtime_device(params: ExperimentConfig) -> torch.device:
    requested = params["device"]
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Parse config, build the requested model, and execute the selected mode."""

    params = get_params()
    ensure_output_dirs(params)
    write_config_snapshot(params)
    set_seed(params["seed"])
    device = _get_runtime_device(params)

    active_model_name = params["target_model"] if params["eval_mode"] == "transfer" else params["model"]
    model = build_model(params, model_name=active_model_name).to(device)

    print(f"Run: {params['run_name']}")
    print(f"Mode: {params['mode']} | Train mode: {params['train_mode']} | Eval mode: {params['eval_mode']}")
    print(f"Device: {device}")

    if params["mode"] == "flops":
        write_flops_summary(params, device)
        return

    train_summary: dict[str, float | int | str] | None = None
    if params["mode"] in {"train", "both"}:
        train_summary = run_training(model, params, device)

    if params["mode"] in {"test", "both"}:
        run_test(model, params, device, train_summary=train_summary)


if __name__ == "__main__":
    main()
