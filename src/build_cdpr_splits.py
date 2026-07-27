from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEEP_PRIORITY = {
    "pediatric_test": 0,
    "pediatric_train": 1,
    "pediatric_supplemental": 2,
    "adult_archive": 3,
    "adult_database": 4,
    "adult_train": 5,
    "adult_test": 6,
}


def allocate_group(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign approximately 60/10/15/15 within an adult source group."""
    indices = frame.index.to_numpy().copy()
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(round(n * 0.60))
    n_cal = int(round(n * 0.10))
    n_val = int(round(n * 0.15))
    labels = (["train"] * n_train + ["calibration"] * n_cal + ["validation"] * n_val)
    labels += ["test"] * (n - len(labels))
    frame.loc[indices, "split"] = labels
    return frame


def build(audit_manifest: Path, output_manifest: Path, output_splits: Path, seed: int) -> None:
    frame = pd.read_csv(audit_manifest)
    frame["priority"] = frame.source_group.map(KEEP_PRIORITY)
    frame = frame.sort_values(["decoded_sha256", "priority", "sample_id"])
    frame["exclusion_reason"] = ""
    frame.loc[frame.foreground_fraction <= 0, "exclusion_reason"] = "empty_mask"

    eligible = frame[frame.exclusion_reason.eq("")]
    duplicate_rows = eligible.duplicated("decoded_sha256", keep="first")
    duplicate_indices = eligible.index[duplicate_rows]
    frame.loc[duplicate_indices, "exclusion_reason"] = "exact_duplicate"
    clean = frame[frame.exclusion_reason.eq("")].copy()
    clean["split"] = ""
    clean["source"] = "panoramic"

    rng = np.random.default_rng(seed)
    pediatric_test = clean.source_group.eq("pediatric_test")
    clean.loc[pediatric_test, "split"] = "test"

    pediatric_pool = clean.source_group.isin(["pediatric_train", "pediatric_supplemental"])
    pediatric_indices = clean.index[pediatric_pool].to_numpy().copy()
    rng.shuffle(pediatric_indices)
    if len(pediatric_indices) != 163:
        raise ValueError(f"Beklenen pediatrik train havuzu 163, bulunan {len(pediatric_indices)}")
    clean.loc[pediatric_indices[:15], "split"] = "validation"
    clean.loc[pediatric_indices[15:35], "split"] = "calibration"
    clean.loc[pediatric_indices[35:], "split"] = "train"

    adult = clean.age_group.eq("adult")
    for group in sorted(clean.loc[adult, "source_group"].unique()):
        group_mask = adult & clean.source_group.eq(group)
        indices = clean.index[group_mask].to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        counts = [int(round(n * 0.60)), int(round(n * 0.10)), int(round(n * 0.15))]
        labels = ["train"] * counts[0] + ["calibration"] * counts[1] + ["validation"] * counts[2]
        labels += ["test"] * (n - len(labels))
        clean.loc[indices, "split"] = labels

    if clean.split.eq("").any():
        raise RuntimeError("Split atanmamış temiz örnekler var.")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id", "patient_id", "source", "source_group", "age_group", "split",
        "image_path", "mask_path", "width", "height", "foreground_fraction",
        "decoded_sha256", "dhash",
    ]
    clean[columns].to_csv(output_manifest, index=False)
    exclusions = frame[frame.exclusion_reason.ne("")][
        ["sample_id", "source_group", "image_path", "mask_path", "exclusion_reason", "decoded_sha256"]
    ]
    exclusions.to_csv(output_manifest.with_name("excluded_samples.csv"), index=False)

    split_payload = {
        name: clean.loc[clean.split.eq(name), "patient_id"].tolist()
        for name in ["train", "calibration", "validation", "test"]
    }
    output_splits.parent.mkdir(parents=True, exist_ok=True)
    output_splits.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    report = {
        "seed": seed,
        "n_clean": len(clean),
        "n_excluded": len(exclusions),
        "excluded_by_reason": exclusions.exclusion_reason.value_counts().to_dict(),
        "split_counts": clean.split.value_counts().to_dict(),
        "split_by_age": clean.groupby(["age_group", "split"]).size().unstack(fill_value=0).to_dict("index"),
        "split_by_source_group": clean.groupby(["source_group", "split"]).size().unstack(fill_value=0).to_dict("index"),
        "leakage_check": {
            "hashes_in_multiple_splits": int((clean.groupby("decoded_sha256").split.nunique() > 1).sum()),
            "patients_in_multiple_splits": int((clean.groupby("patient_id").split.nunique() > 1).sum()),
        },
        "limitation": "patient_id is file-derived because the release contains no separate patient identifier; one radiograph per patient is assumed and must be stated.",
    }
    output_manifest.with_name("split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate audited CDPR bundle and create leakage-safe splits")
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-splits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.audit_manifest, args.output_manifest, args.output_splits, args.seed)


if __name__ == "__main__":
    main()
