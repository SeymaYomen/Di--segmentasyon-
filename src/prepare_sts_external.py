from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def decoded_sha256(image: Image.Image) -> str:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    header = f"{array.shape[0]}x{array.shape[1]}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def perceptual_features(image: Image.Image) -> tuple[int, np.ndarray]:
    gray = image.convert("L")
    dhash_array = np.asarray(
        gray.resize((17, 16), Image.Resampling.LANCZOS), dtype=np.int16
    )
    bits = (dhash_array[:, 1:] >= dhash_array[:, :-1]).reshape(-1)
    dhash = int("".join("1" if bit else "0" for bit in bits), 2)

    signature = np.asarray(
        gray.resize((64, 32), Image.Resampling.LANCZOS), dtype=np.float32
    ).reshape(-1)
    signature -= signature.mean()
    norm = float(np.linalg.norm(signature))
    if norm:
        signature /= norm
    return dhash, signature


def load_reference_features(manifests: list[Path]) -> list[dict]:
    references: list[dict] = []
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Referans manifest bulunamadı: {manifest}")
        frame = pd.read_csv(manifest)
        if "image_path" not in frame:
            raise ValueError(f"image_path sütunu yok: {manifest}")
        for row in frame.itertuples(index=False):
            path = Path(str(row.image_path))
            if not path.exists():
                continue
            with Image.open(path) as image:
                image.load()
                dhash, signature = perceptual_features(image)
                references.append(
                    {
                        "dataset": manifest.stem,
                        "sample_id": str(getattr(row, "sample_id", path.stem)),
                        "decoded_sha256": decoded_sha256(image),
                        "dhash": dhash,
                        "signature": signature,
                    }
                )
    if not references:
        raise RuntimeError(
            "Hiçbir referans görüntü okunamadı. CDPR manifestindeki image_path yollarını "
            "bu çalışma ortamına göre yeniden üretmeden STS dış testini hazırlamayın."
        )
    return references


def overlap_reason(
    digest: str,
    dhash: int,
    signature: np.ndarray,
    references: list[dict],
) -> tuple[str | None, str | None]:
    for reference in references:
        if digest == reference["decoded_sha256"]:
            return "reference_exact_duplicate", reference["sample_id"]

    for reference in references:
        distance = (dhash ^ reference["dhash"]).bit_count()
        if distance > 20:
            continue
        correlation = float(np.dot(signature, reference["signature"]))
        if distance <= 12 and correlation >= 0.97:
            return "reference_perceptual_duplicate", reference["sample_id"]
        if correlation >= 0.93:
            return "reference_ambiguous_similarity", reference["sample_id"]
    return None, None


def prepare(
    raw_root: Path,
    output_manifest: Path,
    output_splits: Path,
    report_path: Path,
    reference_manifests: list[Path],
) -> dict:
    metadata_path = raw_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} bulunamadı; önce src.download_sts_2d çalıştırılmalı."
        )
    metadata = pd.read_csv(metadata_path)
    references = load_reference_features(reference_manifests)
    rows: list[dict] = []
    exclusions: list[dict] = []
    seen_hashes: set[str] = set()

    for row in metadata.itertuples(index=False):
        image_path = Path(str(row.image_path))
        mask_path = Path(str(row.mask_path))
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image.load()
                mask.load()
                if image.size != mask.size:
                    raise ValueError(f"size_mismatch:{image.size}!={mask.size}")
                mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
                foreground = int((mask_array > 0).sum())
                if foreground == 0:
                    raise ValueError("empty_mask")
                digest = decoded_sha256(image)
                if digest in seen_hashes:
                    exclusions.append(
                        {"sample_id": row.sample_id, "reason": "internal_exact_duplicate"}
                    )
                    continue
                dhash, signature = perceptual_features(image)
                reason, match = overlap_reason(digest, dhash, signature, references)
                if reason:
                    exclusions.append(
                        {
                            "sample_id": row.sample_id,
                            "reason": reason,
                            "matched_reference": match,
                        }
                    )
                    continue
                seen_hashes.add(digest)
                rows.append(
                    {
                        "sample_id": f"sts_external:{row.sample_id}",
                        "patient_id": f"sts_external:{row.sample_id}",
                        "source": "panoramic",
                        "source_group": "sts_2d_external_clean",
                        "age_group": row.age_group,
                        "split": "test",
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "width": image.width,
                        "height": image.height,
                        "foreground_fraction": foreground / mask_array.size,
                        "decoded_sha256": digest,
                        "dhash256": f"{dhash:064x}",
                    }
                )
        except Exception as exc:
            exclusions.append(
                {
                    "sample_id": row.sample_id,
                    "reason": f"processing_error:{type(exc).__name__}:{exc}",
                }
            )

    manifest_columns = [
        "sample_id", "patient_id", "source", "source_group", "age_group", "split",
        "image_path", "mask_path", "width", "height", "foreground_fraction",
        "decoded_sha256", "dhash256",
    ]
    frame = pd.DataFrame(rows, columns=manifest_columns)
    excluded = pd.DataFrame(
        exclusions, columns=["sample_id", "reason", "matched_reference"]
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_splits.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_manifest, index=False)
    excluded.to_csv(output_manifest.with_name("excluded_samples.csv"), index=False)
    output_splits.write_text(
        json.dumps(
            {
                "train": [],
                "calibration": [],
                "validation": [],
                "test": frame.patient_id.astype(str).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "dataset": "STS-2D-Tooth",
        "role": "external_test_only_after_overlap_exclusion",
        "n_downloaded_labeled": len(metadata),
        "n_reference_images_checked": len(references),
        "n_clean_external_test": len(frame),
        "n_excluded": len(excluded),
        "clean_by_age": frame.age_group.value_counts().to_dict()
        if not frame.empty
        else {},
        "excluded_by_reason": excluded.reason.value_counts().to_dict()
        if not excluded.empty
        else {},
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/MedOtter/STS-2D-Tooth",
        "warning": (
            "STS contains material derived from the earlier CDPR work. Only samples surviving "
            "exact and perceptual overlap screening may be reported as external test data."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STS-2D-Tooth örneklerini CDPR/Mendeley tekrarlarından arındır"
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw/sts_2d_labeled")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/sts_external/manifest.csv"),
    )
    parser.add_argument(
        "--output-splits", type=Path, default=Path("data/splits/sts_external.json")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/sts_external/audit_summary.json"),
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        action="append",
        required=True,
        help="CDPR ve Mendeley manifestleri için iki kez kullanılabilir.",
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
