"""Feature capture helpers for t-SNE-ready exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models import resolve_classifier_module


class PenultimateFeatureRecorder:
    """Capture the input to the final linear classifier layer."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self.hook = resolve_classifier_module(model).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        del module, output
        self.features = inputs[0].detach()

    def take(self) -> torch.Tensor:
        if self.features is None:
            raise RuntimeError("No features were captured before take() was called.")
        return self.features.flatten(start_dim=1).cpu()

    def close(self) -> None:
        self.hook.remove()


def build_feature_rows(
    features: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sample_kind: str,
    model_name: str,
    run_name: str,
    attack_norm: str,
) -> dict[str, np.ndarray]:
    """Convert one batch of features into numpy arrays for export."""

    sample_count = int(labels.size(0))
    return {
        "features": features.numpy().astype(np.float32),
        "labels": labels.cpu().numpy().astype(np.int64),
        "predictions": predictions.cpu().numpy().astype(np.int64),
        "sample_kind": np.asarray([sample_kind] * sample_count),
        "model_name": np.asarray([model_name] * sample_count),
        "run_name": np.asarray([run_name] * sample_count),
        "attack_norm": np.asarray([attack_norm] * sample_count),
    }


def save_feature_export(path: str | Path, rows: list[dict[str, np.ndarray]]) -> Path:
    """Persist concatenated feature arrays in one ``.npz`` file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot save an empty feature export.")

    merged: dict[str, np.ndarray] = {}
    for key in rows[0]:
        merged[key] = np.concatenate([row[key] for row in rows], axis=0)

    np.savez_compressed(output_path, **merged)
    return output_path
