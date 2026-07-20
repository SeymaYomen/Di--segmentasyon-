import numpy as np
import torch

from src.conformal import finite_sample_quantile, prediction_set
from src.losses import DiceBCELoss
from src.metrics import binary_metrics
from src.preprocessing import split_patch2


def test_patch2_preserves_width_and_alignment():
    image = np.zeros((4, 7, 3), np.uint8); mask = np.zeros((4, 7), np.uint8)
    parts = split_patch2(image, mask)
    assert [p[0].shape[1] for p in parts] == [3, 4]
    assert sum(p[0].shape[1] for p in parts) == image.shape[1]


def test_metrics_perfect_prediction():
    targets = torch.tensor([[[[0.0, 1.0]]]])
    logits = torch.tensor([[[[-20.0, 20.0]]]])
    assert binary_metrics(logits, targets)["dice"] > 0.999
    assert DiceBCELoss()(logits, targets).item() < 0.001


def test_conformal_set_logic():
    q = finite_sample_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.34)
    assert q == 0.3
    inc0, inc1 = prediction_set(np.array([0.1, 0.5, 0.9]), 0.6)
    assert inc0.tolist() == [True, True, False]
    assert inc1.tolist() == [False, True, True]

