"""Reusable adversarial attack helpers for robustness evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _default_step_size(epsilon: float, steps: int, norm: str) -> float:
    if norm == "linf":
        return max(epsilon / 4.0, 1.0 / 255.0)
    return max(epsilon / float(steps), 0.025)


def _project_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factors = torch.minimum(torch.ones_like(norms), torch.full_like(norms, epsilon) / norms)
    return (flat * factors).view_as(delta)


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float = 0.0,
    norm: str = "linf",
    random_start: bool = True,
) -> torch.Tensor:
    """Generate PGD adversarial examples for the provided batch."""

    model.eval()
    alpha = step_size if step_size > 0 else _default_step_size(epsilon, steps, norm)

    if random_start:
        if norm == "linf":
            delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        else:
            delta = torch.randn_like(images)
            delta = _project_l2(delta, epsilon)
    else:
        delta = torch.zeros_like(images)

    adv = (images + delta).clamp(0.0, 1.0).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv)[0]

        if norm == "linf":
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
        elif norm == "l2":
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            normalized_grad = (grad_flat / grad_norm).view_as(grad)
            adv = adv.detach() + alpha * normalized_grad
            delta = _project_l2(adv - images, epsilon)
            adv = images + delta
        else:
            raise ValueError(f"Unsupported PGD norm: {norm}")

        adv = adv.clamp(0.0, 1.0).detach()

    return adv
