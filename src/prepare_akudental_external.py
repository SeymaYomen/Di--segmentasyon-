from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PERMANENT_FDI = {
    *(range(11, 19)),
    *(range(21, 29)),
    *(range(31, 39)),
    *(range(41, 49)),
}


def decoded_sha256(image: Image.Image) -> str:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    header = f"{array.shape[0]}x{array.shape[1]}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def category_kind(name: str) -> str:
    normalized = name.strip().lower()
    match = re.search(r"(?<!\d)(\d{2})(?!\d)", normalized)
    if match and int(match.group(1)) in PERMANENT_FDI:
        return "natural_tooth"
    if "implant" in normalized:
        return "implant"
    return "other"


def find_annotation(raw_root: Path, annotation_json: Path | None) -> Path:
    if annotation_json is not None:
        return annotation_json
    matches = list(raw_root.rglob("akudental_instances.json"))
    if len(matches) != 1:
        raise ValueError(
            "Tam olarak bir akudental_instances.json bulunmalı; "
            f"bulunan dosya sayısı: {len(matches)}"
        )
    return matches[0]


def image_index(raw_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in index:
                duplicates.add(path.name)
            else:
                index[path.name] = path
    for name in duplicates:
        index.pop(name, None)
    return index


def reference_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "decoded_sha256" in frame:
            hashes.update(frame.decoded_sha256.dropna().astype(str))
    return hashes


def polygon_mask(segmentation: object, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32)
            if points.size < 6 or points.size % 2:
                raise ValueError("invalid_polygon")
            vertices = [tuple(point) for point in points.reshape(-1, 2).tolist()]
            draw.polygon(vertices, fill=255)
        return np.asarray(canvas, dtype=np.uint8)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise RuntimeError("RLE maske için pycocotools kurulmalı") from exc
        decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return (decoded > 0).astype(np.uint8) * 255
    raise ValueError("unsupported_segmentation")


def prepare(
    raw_root: Path,
    output_root: Path,
    output_manifest: Path,
    output_splits: Path,
    report_path: Path,
    annotation_json: Path | None = None,
    reference_manifests: list[Path] | None = None,
    implant_policy: str = "exclude",
) -> dict:
    annotation_path = find_annotation(raw_root, annotation_json)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in coco["images"]}
    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)

    indexed_images = image_index(raw_root)
    known_hashes = reference_hashes(reference_manifests or [])
    selected_kinds = {"natural_tooth"}
    if implant_policy == "include":
        selected_kinds.add("implant")

    mask_dir = output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    exclusions: list[dict] = []
    seen_hashes: set[str] = set()
    selected_annotation_count = 0

    for image_id, record in sorted(images.items()):
        file_name = Path(record["file_name"]).name
        image_path = indexed_images.get(file_name)
        if image_path is None:
            exclusions.append({"sample_id": image_id, "reason": f"missing_image:{file_name}"})
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                if record.get("width") and int(record["width"]) != width:
                    raise ValueError("coco_width_mismatch")
                if record.get("height") and int(record["height"]) != height:
                    raise ValueError("coco_height_mismatch")
                digest = decoded_sha256(image)

            if digest in seen_hashes:
                exclusions.append({"sample_id": image_id, "reason": "internal_exact_duplicate"})
                continue
            if digest in known_hashes:
                exclusions.append({"sample_id": image_id, "reason": "reference_exact_duplicate"})
                continue

            combined = np.zeros((height, width), dtype=np.uint8)
            sample_selected = 0
            for annotation in annotations.get(image_id, []):
                name = categories[int(annotation["category_id"])]
                if category_kind(name) not in selected_kinds:
                    continue
                combined = np.maximum(
                    combined,
                    polygon_mask(annotation.get("segmentation"), height, width),
                )
                sample_selected += 1
            if not combined.any():
                exclusions.append({"sample_id": image_id, "reason": "empty_selected_mask"})
                continue

            mask_path = mask_dir / f"{Path(file_name).stem}.png"
            Image.fromarray(combined, mode="L").save(mask_path)
            seen_hashes.add(digest)
            selected_annotation_count += sample_selected
            rows.append(
                {
                    "sample_id": f"akudental:{image_id}",
                    "patient_id": f"akudental:{image_id}",
                    "source": "panoramic",
                    "source_group": "akudental_external",
                    "age_group": "adult",
                    "split": "test",
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "width": width,
                    "height": height,
                    "foreground_fraction": float((combined > 0).mean()),
                    "decoded_sha256": digest,
                }
            )
        except Exception as exc:
            exclusions.append(
                {"sample_id": image_id, "reason": f"processing_error:{type(exc).__name__}:{exc}"}
            )

    frame = pd.DataFrame(rows)
    excluded = pd.DataFrame(exclusions, columns=["sample_id", "reason"])
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

    category_summary = {
        name: category_kind(name) for _, name in sorted(categories.items())
    }
    report = {
        "dataset": "AKUDENTAL",
        "role": "external_test_only",
        "annotation_file": str(annotation_path.resolve()),
        "n_coco_images": len(images),
        "n_usable_images": len(frame),
        "n_excluded": len(excluded),
        "selected_annotations": selected_annotation_count,
        "implant_policy": implant_policy,
        "selected_target": "natural permanent teeth" + (
            " and implants" if implant_policy == "include" else ""
        ),
        "category_classification": category_summary,
        "excluded_by_reason": excluded.reason.value_counts().to_dict()
        if not excluded.empty
        else {},
        "reference_overlap_check": "performed" if known_hashes else "not_available",
        "license": "CC BY-NC-SA 4.0",
        "source": "https://github.com/melihoz/AKUDENTAL",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AKUDENTAL COCO poligonlarını ikili diş maskesine dönüştür ve dış test manifesti üret"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/akudental")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/akudental/manifest.csv"),
    )
    parser.add_argument(
        "--output-splits", type=Path, default=Path("data/splits/akudental.json")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/akudental/audit_summary.json"),
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        action="append",
        default=[],
        help="CDPR ve Mendeley manifestleri için seçenek tekrarlanabilir.",
    )
    parser.add_argument(
        "--implant-policy", choices=["exclude", "include"], default="exclude"
    )
    args = parser.parse_args()
    prepare(
        raw_root=args.raw_root,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        output_splits=args.output_splits,
        report_path=args.report,
        annotation_json=args.annotation_json,
        reference_manifests=args.reference_manifest,
        implant_policy=args.implant_policy,
    )


if __name__ == "__main__":
    main()
