from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import read_pair, resize_pair, save_pair, split_patch2

REQUIRED = {"image_path", "mask_path", "patient_id", "source"}


def patient_split(patient_ids: list[str], seed: int) -> dict[str, list[str]]:
    ids = sorted(set(map(str, patient_ids)))
    if len(ids) < 8:
        raise ValueError("Güvenilir dört yönlü ayrım için en az 8 hasta gerekir.")
    train, rest = train_test_split(ids, test_size=0.40, random_state=seed)
    calibration, rest = train_test_split(rest, test_size=0.75, random_state=seed)
    validation, test = train_test_split(rest, test_size=0.50, random_state=seed)
    return {"train": train, "calibration": calibration, "validation": validation, "test": test}


def run(manifest: Path, output: Path, split_output: Path, size: tuple[int, int], seed: int) -> None:
    frame = pd.read_csv(manifest, dtype={"patient_id": str})
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest sütunları eksik: {sorted(missing)}")
    if not set(frame.source).issubset({"panoramic", "bitewing"}):
        raise ValueError("source yalnızca panoramic veya bitewing olabilir.")

    rows = []
    for row_id, row in frame.reset_index(drop=True).iterrows():
        image, mask = read_pair(row.image_path, row.mask_path)
        parts = split_patch2(image, mask) if row.source == "panoramic" else [(image, mask, "full")]
        for image_part, mask_part, suffix in parts:
            image_part, mask_part = resize_pair(image_part, mask_part, size)
            stem = f"{row_id:06d}_{row.patient_id}_{suffix}"
            image_out = (output / "images" / f"{stem}.png").resolve()
            mask_out = (output / "masks" / f"{stem}.png").resolve()
            save_pair(image_part, mask_part, image_out, mask_out)
            rows.append({"image_path": str(image_out), "mask_path": str(mask_out),
                         "patient_id": str(row.patient_id), "source": row.source})

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    splits = patient_split(frame.patient_id.tolist(), seed)
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_output.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"{len(rows)} örnek hazırlandı; split: {split_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.manifest, args.output, args.split_output, (args.height, args.width), args.seed)


if __name__ == "__main__":
    main()

