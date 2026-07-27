from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


GROUPS = {
    "adult_archive": ("Adult tooth segmentation dataset/Archive/images", "Adult tooth segmentation dataset/Archive/mask", "adult", "unspecified"),
    "adult_train": ("Adult tooth segmentation dataset/Dataset and code/train/images", "Adult tooth segmentation dataset/Dataset and code/train/masks", "adult", "train"),
    "adult_test": ("Adult tooth segmentation dataset/Dataset and code/test/images", "Adult tooth segmentation dataset/Dataset and code/test/masks", "adult", "test"),
    "adult_database": ("Adult tooth segmentation dataset/Panoramic radiography database/images", "Adult tooth segmentation dataset/Panoramic radiography database/mask", "adult", "unspecified"),
    "pediatric_train": ("Children's dental caries segmentation dataset/Train/images", "Children's dental caries segmentation dataset/Train/mask", "pediatric", "train_pool"),
    "pediatric_supplemental": ("Children's dental caries segmentation dataset/Supplemental content-93/images", "Children's dental caries segmentation dataset/Supplemental content-93/mask", "pediatric", "train_pool"),
    "pediatric_test": ("Children's dental caries segmentation dataset/Test/images", "Children's dental caries segmentation dataset/Test/mask", "pediatric", "test"),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalized_stem(path: Path) -> str:
    stem = path.stem
    return stem[:-7] if stem.endswith("_pseudo") else stem


def decoded_sha256(image: Image.Image) -> str:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    header = f"{array.shape[0]}x{array.shape[1]}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def difference_hash(image: Image.Image, width: int = 16, height: int = 16) -> str:
    sample = np.asarray(image.convert("L").resize((width + 1, height), Image.Resampling.LANCZOS))
    bits = (sample[:, 1:] >= sample[:, :-1]).reshape(-1)
    return hex(int("".join("1" if bit else "0" for bit in bits), 2))[2:].zfill(width * height // 4)


def index_directory(directory: Path) -> dict[str, Path]:
    result = {}
    if not directory.exists():
        return result
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = normalized_stem(path)
            if key in result:
                raise ValueError(f"Aynı normalize ada sahip iki dosya var: {result[key]} ve {path}")
            result[key] = path
    return result


def audit(raw_root: Path, output_dir: Path) -> None:
    rows, errors = [], []
    for group, (image_rel, mask_rel, age_group, split_hint) in GROUPS.items():
        images = index_directory(raw_root / image_rel)
        masks = index_directory(raw_root / mask_rel)
        for key in sorted(set(images) | set(masks)):
            image_path, mask_path = images.get(key), masks.get(key)
            if image_path is None or mask_path is None:
                errors.append({"group": group, "key": key, "error": "missing_image" if image_path is None else "missing_mask"})
                continue
            try:
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    image.load(); mask.load()
                    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
                    if image.size != mask.size:
                        errors.append({"group": group, "key": key, "error": f"size_mismatch:{image.size}!={mask.size}"})
                        continue
                    foreground = int((mask_array > 0).sum())
                    if foreground == 0:
                        errors.append({"group": group, "key": key, "error": "empty_mask"})
                    rows.append({
                        "sample_id": f"{group}:{key}", "patient_id": f"{group}:{key}",
                        "source_group": group, "age_group": age_group, "split_hint": split_hint,
                        "image_path": str(image_path.resolve()), "mask_path": str(mask_path.resolve()),
                        "width": image.width, "height": image.height,
                        "foreground_fraction": foreground / mask_array.size,
                        "decoded_sha256": decoded_sha256(image), "dhash": difference_hash(image),
                    })
            except Exception as exc:
                errors.append({"group": group, "key": key, "error": f"read_error:{type(exc).__name__}:{exc}"})

    frame = pd.DataFrame(rows)
    error_frame = pd.DataFrame(errors, columns=["group", "key", "error"])
    if not frame.empty:
        duplicate_counts = frame.groupby("decoded_sha256").sample_id.transform("count")
        frame["exact_duplicate_count"] = duplicate_counts
        frame["is_exact_duplicate"] = duplicate_counts > 1
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "manifest_audited.csv", index=False)
    error_frame.to_csv(output_dir / "audit_errors.csv", index=False)

    duplicate_groups = []
    if not frame.empty:
        for digest, group in frame.groupby("decoded_sha256"):
            if len(group) > 1:
                duplicate_groups.append({"hash": digest, "samples": group.sample_id.tolist()})
    summary = {
        "n_valid_pairs": len(frame), "n_errors": len(error_frame),
        "pairs_by_group": frame.source_group.value_counts().sort_index().to_dict() if not frame.empty else {},
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_samples": sum(len(x["samples"]) for x in duplicate_groups),
        "resolution_counts": {f"{w}x{h}": int(n) for (w, h), n in frame.groupby(["width", "height"]).size().items()} if not frame.empty else {},
        "error_types": dict(Counter(error_frame.error)) if not error_frame.empty else {},
        "notes": [
            "patient_id is provisionally derived from the source group and file stem; replace it if real patient identifiers are provided.",
            "Exact duplicates use SHA-256 over decoded grayscale pixels. dHash is recorded for later near-duplicate analysis.",
            "Pediatric *_pseudo masks were visually verified as whole-tooth segmentation masks; caries COCO annotations are a separate target.",
        ],
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "duplicate_groups.json").write_text(json.dumps(duplicate_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CDPR bundle görüntü/maske kalite ve tekrar denetimi")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw_root, args.output_dir)


if __name__ == "__main__":
    main()
