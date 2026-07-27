from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import device_from_system, load_config, read_splits
from .models import build_model
from .pipeline import make_dataset


def load_trained_model(config: dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = build_model(config["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    model.eval()
    return model


def predict(model: torch.nn.Module, image: torch.Tensor, device: torch.device, threshold: float) -> np.ndarray:
    with torch.inference_mode():
        probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0]
    return (probability.cpu().numpy() >= threshold).astype(np.uint8)


def overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = image.copy()
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    colors = (
        (target & prediction, np.asarray([0, 220, 0])),       # doğru pozitif
        (~target & prediction, np.asarray([235, 55, 55])),    # yanlış pozitif
        (target & ~prediction, np.asarray([255, 165, 0])),    # yanlış negatif
    )
    for selected, color in colors:
        if selected.any():
            result[selected] = (0.55 * result[selected] + 0.45 * color).astype(np.uint8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dış testten kimliksiz iyi/zor örnekleri seçip Baseline-CLAHE karşılaştırma paneli üretir."
    )
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--clahe-config", required=True)
    parser.add_argument("--clahe-checkpoint", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--clahe-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--best-count", type=int, default=2)
    parser.add_argument("--hard-count", type=int, default=2)
    args = parser.parse_args()

    baseline_config = load_config(args.baseline_config)
    clahe_config = load_config(args.clahe_config)
    baseline_splits = read_splits(baseline_config["data"]["split_file"])
    clahe_splits = read_splits(clahe_config["data"]["split_file"])
    baseline_dataset = make_dataset(baseline_config, baseline_splits, "test")
    clahe_dataset = make_dataset(clahe_config, clahe_splits, "test")

    baseline_frame = pd.read_csv(args.baseline_metrics, dtype={"patient_id": str})
    clahe_frame = pd.read_csv(args.clahe_metrics, dtype={"patient_id": str})
    paired = baseline_frame.merge(
        clahe_frame, on="patient_id", suffixes=("_baseline", "_clahe"), validate="one_to_one"
    )
    paired["mean_dice"] = paired[["dice_baseline", "dice_clahe"]].mean(axis=1)
    selected = pd.concat(
        [
            paired.nlargest(args.best_count, "mean_dice").assign(difficulty="İyi"),
            paired.nsmallest(args.hard_count, "mean_dice").assign(difficulty="Zor"),
        ],
        ignore_index=True,
    )

    baseline_index = {str(baseline_dataset[index][3]): index for index in range(len(baseline_dataset))}
    clahe_index = {str(clahe_dataset[index][3]): index for index in range(len(clahe_dataset))}
    if not set(selected.patient_id).issubset(baseline_index.keys() & clahe_index.keys()):
        raise RuntimeError("Seçilen örnekler iki değerlendirme veri kümesinde birebir eşleşmiyor.")

    device = device_from_system()
    baseline_model = load_trained_model(baseline_config, args.baseline_checkpoint, device)
    clahe_model = load_trained_model(clahe_config, args.clahe_checkpoint, device)
    rows = len(selected)
    figure, axes = plt.subplots(rows, 4, figsize=(15, 3.8 * rows), squeeze=False)

    public_rows = []
    for row_index, record in selected.iterrows():
        patient_id = str(record.patient_id)
        baseline_image, baseline_mask, _, _ = baseline_dataset[baseline_index[patient_id]]
        clahe_image, clahe_mask, _, _ = clahe_dataset[clahe_index[patient_id]]
        target = baseline_mask[0].numpy() >= 0.5
        if not np.array_equal(target, clahe_mask[0].numpy() >= 0.5):
            raise RuntimeError("Baseline ve CLAHE hedef maskeleri eşleşmiyor.")

        baseline_prediction = predict(
            baseline_model,
            baseline_image,
            device,
            float(baseline_config["training"].get("threshold", 0.5)),
        )
        clahe_prediction = predict(
            clahe_model,
            clahe_image,
            device,
            float(clahe_config["training"].get("threshold", 0.5)),
        )
        baseline_rgb = np.clip(baseline_image.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        clahe_rgb = np.clip(clahe_image.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        anonymous_id = f"{record.difficulty} {row_index + 1}"

        axes[row_index, 0].imshow(baseline_rgb)
        axes[row_index, 0].set_title(f"{anonymous_id} — Girdi")
        axes[row_index, 1].imshow(target, cmap="gray")
        axes[row_index, 1].set_title("Gerçek maske")
        axes[row_index, 2].imshow(overlay(baseline_rgb, target, baseline_prediction))
        axes[row_index, 2].set_title(f"Baseline (Dice={record.dice_baseline:.3f})")
        axes[row_index, 3].imshow(overlay(clahe_rgb, target, clahe_prediction))
        axes[row_index, 3].set_title(f"CLAHE (Dice={record.dice_clahe:.3f})")
        for axis in axes[row_index]:
            axis.axis("off")

        public_rows.append(
            {
                "example": anonymous_id,
                "difficulty": record.difficulty,
                "baseline_dice": float(record.dice_baseline),
                "clahe_dice": float(record.dice_clahe),
            }
        )

    figure.suptitle(
        "Bağımsız dış OPG testinde iyi ve zor örnekler\n"
        "Yeşil: doğru pozitif, kırmızı: yanlış pozitif, turuncu: yanlış negatif",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(public_rows).to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "selection": "İki yöntemin ortalama Dice değerine göre en yüksek ve en düşük örnekler.",
                "patient_identifiers_published": False,
                "rows": public_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Nitel panel kaydedildi: {output}")


if __name__ == "__main__":
    main()
