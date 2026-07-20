from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import device_from_system, load_config, read_splits
from .models import build_model
from .pipeline import make_dataset, make_eval_loader


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores).reshape(-1)
    if scores.size == 0 or not 0 < alpha < 1:
        raise ValueError("Skorlar boş olmamalı ve alpha 0-1 arasında olmalıdır.")
    level = min(1.0, np.ceil((scores.size + 1) * (1 - alpha)) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


def prediction_set(probabilities: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    include_background = probabilities <= threshold
    include_tooth = (1.0 - probabilities) <= threshold
    return include_background, include_tooth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(); config = load_config(args.config)
    splits = read_splits(config["data"]["split_file"]); device = device_from_system()
    model = build_model(config["model"]).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"] if "model" in payload else payload); model.eval()
    rng = np.random.default_rng(config["seed"]); scores = []
    cal_loader = make_eval_loader(make_dataset(config, splits, "calibration"), config)
    limit = config["conformal"].get("max_pixels_per_image", 10000)
    with torch.no_grad():
        for images, masks, _, _ in cal_loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            targets = masks.numpy()
            batch_scores = np.where(targets == 1, 1 - probs, probs).reshape(len(images), -1)
            for row in batch_scores:
                scores.append(rng.choice(row, size=min(limit, row.size), replace=False))
    threshold = finite_sample_quantile(np.concatenate(scores), config["conformal"]["alpha"])
    test_loader = make_eval_loader(make_dataset(config, splits, "test"), config)
    example_dir = Path(config["results_dir"]) / "conformal_examples"; example_dir.mkdir(parents=True, exist_ok=True)
    total = covered = ambiguous = 0
    with torch.no_grad():
        for batch_id, (images, masks, _, _) in enumerate(test_loader):
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            targets = masks.numpy().astype(bool)
            inc0, inc1 = prediction_set(probs, threshold)
            covered += np.where(targets, inc1, inc0).sum(); total += targets.size
            ambiguous += (inc0 == inc1).sum()
            if batch_id == 0:
                for i in range(min(4, len(images))):
                    gray = (images[i].mean(0).numpy() * 255).astype(np.uint8)
                    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    overlay[(inc0[i, 0] == inc1[i, 0])] = (0, 0, 255)
                    cv2.imwrite(str(example_dir / f"example_{i}.png"), overlay)
    report = {"alpha": config["conformal"]["alpha"], "threshold": threshold,
              "empirical_pixel_coverage": float(covered / total),
              "ambiguous_or_empty_fraction": float(ambiguous / total),
              "note": "Pixel sampling gives marginal pixel coverage; spatial dependence limits the guarantee."}
    output = Path(config["results_dir"]) / "conformal.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

