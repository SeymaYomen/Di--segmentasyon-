from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import device_from_system, load_config, read_splits
from .models import build_model
from .pipeline import make_dataset, make_eval_loader


def blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.5) -> None:
    if not mask.any():
        return
    image[mask] = ((1 - alpha) * image[mask] + alpha * np.asarray(color)).astype(np.uint8)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground truth ve model tahminlerini renkli bindirme olarak kaydet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    splits = read_splits(config["data"]["split_file"])
    loader = make_eval_loader(make_dataset(config, splits, "test"), config)
    device = device_from_system()
    model = build_model(config["model"]).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()

    output_dir = Path(config["results_dir"]) / "prediction_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    threshold = float(config["training"].get("threshold", 0.5))
    with torch.no_grad():
        for images, masks, _, patients in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
            for index in range(len(images)):
                rgb = np.clip(images[index].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                target = masks[index, 0].numpy() >= 0.5
                prediction = probabilities[index, 0] >= threshold
                true_positive = target & prediction
                false_positive = ~target & prediction
                false_negative = target & ~prediction
                overlay = rgb.copy()
                blend(overlay, true_positive, (0, 255, 0))
                blend(overlay, false_positive, (255, 0, 0))
                blend(overlay, false_negative, (0, 100, 255))
                cv2.imwrite(
                    str(output_dir / f"{written + 1:02d}_{safe_name(str(patients[index]))}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                written += 1
                if written >= args.count:
                    print(f"{written} örnek kaydedildi: {output_dir}")
                    return
    print(f"{written} örnek kaydedildi: {output_dir}")


if __name__ == "__main__":
    main()
