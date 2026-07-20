from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .common import device_from_system, load_config, read_splits
from .metrics import binary_metrics
from .models import build_model
from .pipeline import make_dataset, make_eval_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    splits = read_splits(config["data"]["split_file"])
    loader = make_eval_loader(make_dataset(config, splits, "test"), config)
    device = device_from_system()
    model = build_model(config["model"]).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval(); rows = []
    with torch.no_grad():
        for images, masks, sources, patients in loader:
            logits = model(images.to(device)).cpu()
            for i in range(len(images)):
                rows.append({"patient_id": patients[i], "source": sources[i],
                             **binary_metrics(logits[i:i+1], masks[i:i+1], config["training"]["threshold"])})
    frame = pd.DataFrame(rows)
    results = Path(config["results_dir"]); results.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results / "per_image_metrics.csv", index=False)
    report = {"overall": frame[["dice", "iou", "pixel_accuracy"]].mean().to_dict(),
              "by_source": frame.groupby("source")[["dice", "iou", "pixel_accuracy"]].mean().to_dict("index"),
              "n_images": len(frame)}
    (results / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

