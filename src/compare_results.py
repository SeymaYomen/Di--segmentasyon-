from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["dice", "iou", "pixel_accuracy"]


def bootstrap_comparison(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> dict:
    rng = np.random.default_rng(seed)
    report: dict[str, dict] = {}
    for metric in METRICS:
        a = internal[metric].to_numpy(dtype=float)
        b = external[metric].to_numpy(dtype=float)
        a_means = rng.choice(a, size=(n_bootstrap, len(a)), replace=True).mean(axis=1)
        b_means = rng.choice(b, size=(n_bootstrap, len(b)), replace=True).mean(axis=1)
        gaps = b_means - a_means
        report[metric] = {
            "internal_mean": float(a.mean()),
            "external_mean": float(b.mean()),
            "external_minus_internal": float(b.mean() - a.mean()),
            "gap_ci95_low": float(np.quantile(gaps, 0.025)),
            "gap_ci95_high": float(np.quantile(gaps, 0.975)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="İç ve dış test performansını bootstrap güven aralığıyla karşılaştır")
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/internal_external_comparison"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    internal = pd.read_csv(args.internal)
    external = pd.read_csv(args.external)
    missing = set(METRICS) - set(internal.columns) | (set(METRICS) - set(external.columns))
    if missing:
        raise ValueError(f"Metrik sütunları eksik: {sorted(missing)}")
    if internal.empty or external.empty:
        raise ValueError("İç ve dış test tabloları boş olmamalıdır.")

    report = bootstrap_comparison(internal, external, args.seed, args.bootstrap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame.from_dict(report, orient="index").rename_axis("metric").to_csv(
        args.output_dir / "comparison.csv"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
