from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); rng = np.random.default_rng(args.seed)
    if args.patients < 8:
        raise ValueError("En az 8 dummy hasta üretin.")
    images = args.output / "images"; masks = args.output / "masks"
    images.mkdir(parents=True, exist_ok=True); masks.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.patients):
        source = "bitewing" if i % 4 == 0 else "panoramic"
        h, w = (192, 256) if source == "bitewing" else (192, 384)
        image = np.clip(rng.normal(55, 12, (h, w)), 0, 255).astype(np.uint8)
        mask = np.zeros((h, w), np.uint8)
        for x in np.linspace(w * .15, w * .85, 8).astype(int):
            center = (int(x + rng.integers(-5, 6)), int(h * .48 + rng.integers(-8, 9)))
            axes = (int(w * .035), int(h * .20))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        image = np.clip(image + (mask > 0) * rng.integers(80, 130), 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_path = (images / f"P{i:03d}.png").resolve(); mask_path = (masks / f"P{i:03d}.png").resolve()
        cv2.imwrite(str(image_path), image); cv2.imwrite(str(mask_path), mask)
        rows.append({"image_path": str(image_path), "mask_path": str(mask_path),
                     "patient_id": f"P{i:03d}", "source": source})
    pd.DataFrame(rows).to_csv(args.output / "manifest.csv", index=False)
    print(f"{args.patients} dummy hasta üretildi: {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()

