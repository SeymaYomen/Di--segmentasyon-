import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.prepare_akudental_external import category_kind, prepare


def make_fixture(root: Path) -> Path:
    image_dir = root / "AKUDENTAL" / "images"
    image_dir.mkdir(parents=True)
    Image.fromarray(np.full((8, 10, 3), 120, dtype=np.uint8)).save(image_dir / "case.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "case.jpg", "width": 10, "height": 8}],
        "categories": [
            {"id": 1, "name": "11"},
            {"id": 33, "name": "Bridge"},
            {"id": 35, "name": "Implant"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]},
            {"id": 2, "image_id": 1, "category_id": 33,
             "segmentation": [[5, 1, 6, 1, 6, 2, 5, 2]]},
            {"id": 3, "image_id": 1, "category_id": 35,
             "segmentation": [[7, 1, 8, 1, 8, 2, 7, 2]]},
        ],
    }
    annotation = root / "AKUDENTAL" / "akudental_instances.json"
    annotation.write_text(json.dumps(coco), encoding="utf-8")
    return annotation


def run_prepare(root: Path, implant_policy: str = "exclude") -> tuple[dict, np.ndarray]:
    report = prepare(
        raw_root=root,
        output_root=root / "processed",
        output_manifest=root / "processed" / "manifest.csv",
        output_splits=root / "splits.json",
        report_path=root / "processed" / "report.json",
        implant_policy=implant_policy,
    )
    manifest = pd.read_csv(root / "processed" / "manifest.csv")
    mask = np.asarray(Image.open(manifest.loc[0, "mask_path"]).convert("L"))
    return report, mask


def test_category_mapping():
    assert category_kind("11") == "natural_tooth"
    assert category_kind("Tooth 48") == "natural_tooth"
    assert category_kind("Implant") == "implant"
    assert category_kind("Filling-Crown") == "other"


def test_default_mask_excludes_restoration_and_implant(tmp_path):
    make_fixture(tmp_path)
    report, mask = run_prepare(tmp_path)
    assert report["n_usable_images"] == 1
    assert mask[2, 2] == 255
    assert mask[1, 5] == 0
    assert mask[1, 7] == 0


def test_implant_can_be_included_explicitly(tmp_path):
    make_fixture(tmp_path)
    _, mask = run_prepare(tmp_path, implant_policy="include")
    assert mask[1, 7] == 255
