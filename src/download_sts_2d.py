from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


DATASET_ID = "MedOtter/STS-2D-Tooth"
LABELED_SPLITS = ("a_pxi_labeled", "c_pxi_labeled")


def save_binary_mask(mask: Image.Image, path: Path) -> None:
    binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    binary.save(path, format="PNG")


def download(output_root: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Önce `pip install datasets` komutunu çalıştırın."
        ) from exc

    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for split_name in LABELED_SPLITS:
        dataset = load_dataset(DATASET_ID, split=split_name)
        for row in dataset:
            sample_id = str(row["sample_id"])
            age_group = "adult" if split_name.startswith("a_") else "pediatric"
            image_path = image_dir / f"{sample_id}.png"
            mask_path = mask_dir / f"{sample_id}.png"
            row["image"].convert("RGB").save(image_path, format="PNG")
            if row["mask"] is None:
                raise ValueError(f"Etiketli bölümde maske eksik: {sample_id}")
            save_binary_mask(row["mask"], mask_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "age_group": age_group,
                    "hf_split": split_name,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                }
            )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_root / "metadata.csv", index=False)
    report = {
        "dataset": "STS-2D-Tooth",
        "source": DATASET_ID,
        "downloaded_labeled_images": len(metadata),
        "by_age": metadata.age_group.value_counts().to_dict(),
        "license": "CC BY 4.0",
        "note": "Only the 900 labeled 2D panoramic samples are downloaded.",
    }
    (output_root / "download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STS-2D-Tooth veri setinin yalnızca 900 maskeli panoramik örneğini indir"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw/sts_2d_labeled")
    )
    args = parser.parse_args()
    download(args.output_root)


if __name__ == "__main__":
    main()
