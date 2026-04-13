"""Reusable Grad-CAM utility for exported clean/adversarial sample pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models import build_model, resolve_last_conv_module


def _load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def _dataset_stats(dataset: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset == "mnist":
        return (0.1307,), (0.3081,)
    return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def denormalize(image: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """Convert a normalized tensor into an ``H x W x C`` numpy image."""

    image = image.detach().cpu()
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    image = image * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0)
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    return image.permute(1, 2, 0).numpy()


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a jet heatmap onto an RGB image."""

    colorized = cm.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colorized, 0.0, 1.0)


class GradCAM:
    """Compute Grad-CAM heatmaps for a given model and convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output) -> None:
        del module, inputs
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        del module, grad_input
        self.gradients = grad_output[0].detach()

    def __call__(self, inputs: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[:, class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        height, width = inputs.shape[2], inputs.shape[3]
        heatmap = Image.fromarray(np.uint8(cam * 255)).resize((width, height), Image.BILINEAR)
        return np.asarray(heatmap, dtype=np.float32) / 255.0, class_idx

    def close(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()


def build_argparser() -> argparse.ArgumentParser:
    """Create a small dedicated CLI for Grad-CAM rendering."""

    parser = argparse.ArgumentParser(description="Generate Grad-CAM figures for saved PGD samples.")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="resnet")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--samples_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/figures")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--transfer_mode",
        choices=["none", "resize_freeze", "modify_finetune"],
        default="none",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sample_limit", type=int, default=4)
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
    )
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    return parser


def render_gradcam_panels(args: argparse.Namespace) -> list[Path]:
    """Load saved samples, generate Grad-CAM panels, and write PNG outputs."""

    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "model": args.model,
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "feature_size": 3072 if args.dataset == "cifar10" else 784,
        "hidden_sizes": [512, 256, 128],
        "dropout": 0.3,
        "vgg_depth": args.vgg_depth,
        "resnet_layers": args.resnet_layers,
        "pretrained": args.pretrained,
        "transfer_mode": args.transfer_mode,
        "freeze_backbone": args.freeze_backbone,
    }
    model = build_model(
        params,
        pretrained=args.pretrained,
        transfer_mode=args.transfer_mode,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    _load_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    target_layer = resolve_last_conv_module(model)
    gradcam = GradCAM(model, target_layer)
    mean, std = _dataset_stats(args.dataset)
    samples = torch.load(args.samples_path, map_location="cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for index, sample in enumerate(samples[: args.sample_limit]):
        clean_tensor = sample["clean_image"].unsqueeze(0).to(device)
        adv_tensor = sample["adv_image"].unsqueeze(0).to(device)

        clean_heatmap, _ = gradcam(clean_tensor, class_idx=sample["clean_prediction"])
        adv_heatmap, _ = gradcam(adv_tensor, class_idx=sample["adv_prediction"])

        clean_image = denormalize(clean_tensor.squeeze(0), mean, std)
        adv_image = denormalize(adv_tensor.squeeze(0), mean, std)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes[0, 0].imshow(clean_image)
        axes[0, 0].set_title(f"Clean | y={sample['true_label']} pred={sample['clean_prediction']}")
        axes[0, 1].imshow(clean_heatmap, cmap="jet")
        axes[0, 1].set_title("Clean Heatmap")
        axes[0, 2].imshow(overlay(clean_image, clean_heatmap))
        axes[0, 2].set_title("Clean Overlay")

        axes[1, 0].imshow(adv_image)
        axes[1, 0].set_title(f"Adversarial | pred={sample['adv_prediction']}")
        axes[1, 1].imshow(adv_heatmap, cmap="jet")
        axes[1, 1].set_title("Adversarial Heatmap")
        axes[1, 2].imshow(overlay(adv_image, adv_heatmap))
        axes[1, 2].set_title("Adversarial Overlay")

        for axis in axes.ravel():
            axis.axis("off")

        plt.tight_layout()
        run_name = args.run_name or sample.get("run_name", Path(args.checkpoint_path).stem)
        output_path = output_dir / f"{run_name}_gradcam_{index:02d}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    gradcam.close()
    return output_paths


def main() -> None:
    """CLI entry point."""

    args = build_argparser().parse_args()
    output_paths = render_gradcam_panels(args)
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
