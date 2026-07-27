from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalized_stem(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_mask") else stem


def index_images(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = normalized_stem(path)
        if key in result:
            raise ValueError(f"Aynı örnek adına sahip iki dosya var: {result[key]} ve {path}")
        result[key] = path
    return result


def find_dataset_dirs(raw_root: Path) -> tuple[Path, Path]:
    image_dirs = [p for p in raw_root.rglob("images") if p.is_dir()]
    mask_dirs = [p for p in raw_root.rglob("masks") if p.is_dir()]
    if len(image_dirs) != 1 or len(mask_dirs) != 1:
        raise ValueError(
            "Tam olarak bir images ve bir masks klasörü bulunmalı; "
            f"bulunan: images={len(image_dirs)}, masks={len(mask_dirs)}"
        )
    return image_dirs[0], mask_dirs[0]


def decoded_sha256(image: Image.Image) -> str:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    header = f"{array.shape[0]}x{array.shape[1]}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def reference_hashes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "decoded_sha256" not in frame.columns:
        return set()
    return set(frame.decoded_sha256.dropna().astype(str))


def prepare(
    raw_root: Path,
    output_manifest: Path,
    output_splits: Path,
    report_path: Path,
    reference_manifest: Path | None = None,
) -> dict:
    image_dir, mask_dir = find_dataset_dirs(raw_root)
    images = index_images(image_dir)
    masks = index_images(mask_dir)
    keys = sorted(set(images) | set(masks))
    known_hashes = reference_hashes(reference_manifest)

    rows: list[dict] = []
    exclusions: list[dict] = []
    seen_hashes: set[str] = set()

    for key in keys:
        image_path, mask_path = images.get(key), masks.get(key)
        if image_path is None or mask_path is None:
            exclusions.append(
                {
                    "sample_id": key,
                    "reason": "missing_image" if image_path is None else "missing_mask",
                }
            )
            continue
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image.load()
                mask.load()
                if image.size != mask.size:
                    exclusions.append(
                        {"sample_id": key, "reason": f"size_mismatch:{image.size}!={mask.size}"}
                    )
                    continue
                mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
                foreground = int((mask_array > 0).sum())
                if foreground == 0:
                    exclusions.append({"sample_id": key, "reason": "empty_mask"})
                    continue
                digest = decoded_sha256(image)
                if digest in seen_hashes:
                    exclusions.append({"sample_id": key, "reason": "internal_exact_duplicate"})
                    continue
                if digest in known_hashes:
                    exclusions.append({"sample_id": key, "reason": "reference_exact_duplicate"})
                    continue
                seen_hashes.add(digest)
                rows.append(
                    {
                        "sample_id": f"opg_external:{key}",
                        "patient_id": f"opg_external:{key}",
                        "source": "panoramic",
                        "source_group": "opg_dentalseg_external",
                        "age_group": "mixed",
                        "split": "test",
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "width": image.width,
                        "height": image.height,
                        "foreground_fraction": foreground / mask_array.size,
                        "decoded_sha256": digest,
                    }
                )
        except Exception as exc:
            exclusions.append(
                {"sample_id": key, "reason": f"read_error:{type(exc).__name__}:{exc}"}
            )

    frame = pd.DataFrame(rows)
    exclusion_frame = pd.DataFrame(exclusions, columns=["sample_id", "reason"])
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_splits.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_manifest, index=False)
    exclusion_frame.to_csv(output_manifest.with_name("excluded_samples.csv"), index=False)

    split_payload = {
        "train": [],
        "calibration": [],
        "validation": [],
        "test": frame.patient_id.astype(str).tolist(),
    }
    output_splits.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    report = {
        "dataset": "OPG-DentalSeg",
        "role": "external_test_only",
        "n_images_found": len(images),
        "n_masks_found": len(masks),
        "n_usable_pairs": len(frame),
        "n_excluded": len(exclusion_frame),
        "excluded_by_reason": exclusion_frame.reason.value_counts().to_dict()
        if not exclusion_frame.empty
        else {},
        "reference_overlap_check": "performed"
        if reference_manifest is not None and reference_manifest.exists() and known_hashes
        else "not_available",
        "resolution_counts": {
            f"{width}x{height}": int(count)
            for (width, height), count in frame.groupby(["width", "height"]).size().items()
        }
        if not frame.empty
        else {},
        "patient_id_note": (
            "The release describes 329 radiographs from 329 patients; file-derived IDs are therefore "
            "used as patient IDs."
        ),
        "license": "CC BY 4.0",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OPG-DentalSeg görüntü-maskelerini denetle ve bağımsız dış test manifesti üret"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-manifest", type=Path, default=Path("data/processed/opg_external/manifest.csv")
    )
    parser.add_argument(
        "--output-splits", type=Path, default=Path("data/splits/opg_external.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/opg_external/audit_summary.json")
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("data/processed/cdpr/manifest.csv"),
        help="Varsa eğitim verisiyle kesin piksel tekrarlarını dışlamak için kullanılır.",
    )
    args = parser.parse_args()
    prepare(
        args.raw_root,
        args.output_manifest,
        args.output_splits,
        args.report,
        args.reference_manifest,
    )


if __name__ == "__main__":
    main()
