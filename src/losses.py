import torch
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return self.alpha * dice_loss + (1 - self.alpha) * bce

