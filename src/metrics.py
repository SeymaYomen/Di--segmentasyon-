from __future__ import annotations

import torch


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = (targets >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    pred_sum, target_sum = pred.sum(dim=dims), target.sum(dim=dims)
    union = pred_sum + target_sum - intersection
    eps = 1e-6
    return {
        "dice": ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item(),
        "iou": ((intersection + eps) / (union + eps)).mean().item(),
        "pixel_accuracy": (pred == target).float().flatten(1).mean(1).mean().item(),
    }

