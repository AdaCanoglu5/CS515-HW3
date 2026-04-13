"""Argument parsing and lightweight experiment configuration for HW3."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _str2bool(value: str | bool) -> bool:
    """Parse permissive boolean CLI values."""

    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _float_or_fraction(value: str) -> float:
    """Parse either a float literal or a simple fraction like ``4/255``."""

    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


@dataclass
class ExperimentConfig:
    """Dataclass-backed config with dict-style compatibility."""

    dataset: str
    data_dir: str
    cifar10c_dir: str
    num_workers: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    model: str
    source_model: str
    target_model: str
    teacher_model: str
    student_model: str
    feature_size: int
    input_size: int
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    dropout: float = 0.3
    vgg_depth: str = "16"
    resnet_layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    mode: str = "both"
    train_mode: str = "clean_finetune"
    eval_mode: str = "clean"
    attack_type: str = "pgd"
    attack_norm: str = "linf"
    attack_epsilon: float = 4.0 / 255.0
    attack_steps: int = 20
    attack_step_size: float = 0.0
    attack_random_start: bool = True
    teacher_checkpoint: str = ""
    student_checkpoint: str = ""
    checkpoint_path: str = ""
    source_checkpoint: str = ""
    target_checkpoint: str = ""
    temperature: float = 4.0
    alpha: float = 0.7
    label_smoothing: float = 0.0
    augmix_enabled: bool = False
    augmix_severity: int = 3
    augmix_width: int = 3
    augmix_depth: int = -1
    augmix_alpha: float = 1.0
    run_name: str = "default_run"
    output_dir: str = "artifacts"
    checkpoint_dir: str = "artifacts/checkpoints"
    run_dir: str = "artifacts/runs"
    robustness_dir: str = "artifacts/robustness"
    features_dir: str = "artifacts/features"
    figures_dir: str = "artifacts/figures"
    config_dir: str = "artifacts/configs"
    save_path: str = "artifacts/checkpoints/default_run.pth"
    csv_path: str = "artifacts/runs/default_run.csv"
    summary_path: str = "artifacts/summary.csv"
    flops_summary_path: str = "artifacts/flops_summary.csv"
    config_path: str = "artifacts/configs/default_run.json"
    val_split: float = 0.1
    log_interval: int = 0
    save_csv: bool = True
    save_config: bool = True
    export_features: bool = False
    feature_export_limit: int = 0
    gradcam_sample_count: int = 8
    cifar10c_severities: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_run_name(args: argparse.Namespace) -> str:
    """Build a deterministic run name when none is provided."""

    if args.mode in {"train", "both"}:
        if args.train_mode == "distill":
            teacher_tag = Path(args.teacher_checkpoint).stem if args.teacher_checkpoint else args.teacher_model
            return f"{args.model}_distill_from_{teacher_tag or 'teacher'}"
        if args.train_mode == "augmix_finetune":
            return f"{args.model}_augmix_teacher"
        return f"{args.model}_clean_teacher"

    if args.eval_mode == "transfer":
        source_tag = Path(args.source_checkpoint).stem if args.source_checkpoint else args.source_model
        target_tag = Path(args.target_checkpoint or args.student_checkpoint).stem if (args.target_checkpoint or args.student_checkpoint) else args.target_model
        return f"transfer_{source_tag or 'source'}_to_{target_tag or 'target'}"

    suffix = args.eval_mode
    if args.eval_mode in {"pgd", "transfer"}:
        suffix = f"{suffix}_{args.attack_norm}"
    return f"{args.model}_{suffix}"


def get_params() -> ExperimentConfig:
    """Parse CLI arguments into a single experiment config."""

    parser = argparse.ArgumentParser(description="CS515 HW3 experiment runner")

    parser.add_argument("--mode", choices=["train", "test", "both", "flops"], default="both")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="resnet")
    parser.add_argument("--source_model", choices=["", "mlp", "cnn", "vgg", "resnet", "mobilenet"], default="")
    parser.add_argument("--target_model", choices=["", "mlp", "cnn", "vgg", "resnet", "mobilenet"], default="")
    parser.add_argument("--teacher_model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="resnet")
    parser.add_argument("--student_model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="cnn")
    parser.add_argument(
        "--train_mode",
        choices=["clean_finetune", "augmix_finetune", "distill"],
        default="clean_finetune",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["clean", "cifar10c", "pgd", "transfer"],
        default="clean",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cifar10c_dir", type=str, default="./data/CIFAR-10-C")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--save_csv", type=_str2bool, default=True)
    parser.add_argument("--save_config", type=_str2bool, default=True)
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18).",
    )

    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--student_checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--source_checkpoint", type=str, default="")
    parser.add_argument("--target_checkpoint", type=str, default="")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--attack_type", choices=["pgd"], default="pgd")
    parser.add_argument("--attack_norm", choices=["linf", "l2"], default="linf")
    parser.add_argument("--attack_epsilon", type=_float_or_fraction, default=4.0 / 255.0)
    parser.add_argument("--attack_steps", type=int, default=20)
    parser.add_argument("--attack_step_size", type=_float_or_fraction, default=0.0)
    parser.add_argument("--attack_random_start", type=_str2bool, default=True)

    parser.add_argument("--augmix_enabled", type=_str2bool, default=False)
    parser.add_argument("--augmix_severity", type=int, default=3)
    parser.add_argument("--augmix_width", type=int, default=3)
    parser.add_argument("--augmix_depth", type=int, default=-1)
    parser.add_argument("--augmix_alpha", type=float, default=1.0)

    parser.add_argument("--export_features", type=_str2bool, default=False)
    parser.add_argument("--feature_export_limit", type=int, default=0)
    parser.add_argument("--gradcam_sample_count", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument(
        "--cifar10c_severities",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5],
        help="CIFAR-10-C severities to evaluate.",
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        feature_size = 784
        input_size = 28
        mean, std = (0.1307,), (0.3081,)
        num_classes = 10
    else:
        feature_size = 3072
        input_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10

    if args.train_mode == "augmix_finetune":
        args.augmix_enabled = True

    source_model = args.source_model or args.teacher_model
    target_model = args.target_model or args.model
    run_name = args.run_name or _default_run_name(args)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    run_dir = output_dir / "runs"
    robustness_dir = output_dir / "robustness"
    features_dir = output_dir / "features"
    figures_dir = output_dir / "figures"
    config_dir = output_dir / "configs"

    return ExperimentConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        cifar10c_dir=args.cifar10c_dir,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        model=args.model,
        source_model=source_model,
        target_model=target_model,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        feature_size=feature_size,
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        num_classes=num_classes,
        dropout=0.3,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        seed=args.seed,
        device=args.device,
        mode=args.mode,
        train_mode=args.train_mode,
        eval_mode=args.eval_mode,
        attack_type=args.attack_type,
        attack_norm=args.attack_norm,
        attack_epsilon=args.attack_epsilon,
        attack_steps=args.attack_steps,
        attack_step_size=args.attack_step_size,
        attack_random_start=args.attack_random_start,
        teacher_checkpoint=args.teacher_checkpoint,
        student_checkpoint=args.student_checkpoint,
        checkpoint_path=args.checkpoint_path,
        source_checkpoint=args.source_checkpoint,
        target_checkpoint=args.target_checkpoint,
        temperature=args.temperature,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        augmix_enabled=args.augmix_enabled,
        augmix_severity=args.augmix_severity,
        augmix_width=args.augmix_width,
        augmix_depth=args.augmix_depth,
        augmix_alpha=args.augmix_alpha,
        run_name=run_name,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        run_dir=str(run_dir),
        robustness_dir=str(robustness_dir),
        features_dir=str(features_dir),
        figures_dir=str(figures_dir),
        config_dir=str(config_dir),
        save_path=str(checkpoint_dir / f"{run_name}.pth"),
        csv_path=str(run_dir / f"{run_name}.csv"),
        summary_path=str(output_dir / "summary.csv"),
        flops_summary_path=str(output_dir / "flops_summary.csv"),
        config_path=str(config_dir / f"{run_name}.json"),
        val_split=args.val_split,
        log_interval=args.log_interval,
        save_csv=args.save_csv,
        save_config=args.save_config,
        export_features=args.export_features,
        feature_export_limit=args.feature_export_limit,
        gradcam_sample_count=args.gradcam_sample_count,
        cifar10c_severities=args.cifar10c_severities,
    )
