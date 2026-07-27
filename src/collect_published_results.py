from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("Internal CDPR", "Baseline", "results/cdpr_baseline/metrics.json"),
    ("Internal CDPR", "CLAHE", "results/cdpr_clahe/metrics.json"),
    ("External OPG", "Baseline", "results/opg_external_baseline/metrics.json"),
    ("External OPG", "CLAHE", "results/opg_external_clahe/metrics.json"),
    ("Clean STS exploratory", "Baseline", "results/sts_external_baseline/metrics.json"),
    ("Clean STS exploratory", "CLAHE", "results/sts_external_clahe/metrics.json"),
]


def main() -> None:
    rows = []
    missing = []
    for dataset, method, source in EXPERIMENTS:
        path = Path(source)
        if not path.exists():
            missing.append(source)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = payload["overall"]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_images": payload["n_images"],
                "dice": overall["dice"],
                "iou": overall["iou"],
                "pixel_accuracy": overall["pixel_accuracy"],
                "source_metrics": source,
                "source_provenance": str(path.parent / "evaluation_provenance.json"),
            }
        )

    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Yayımlanmış sonuç tablosu oluşturulmadı. Eksik ham değerlendirme "
            f"dosyaları:\n{formatted}"
        )

    output = Path("results/published/final_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} deney kaydı yazıldı: {output}")


if __name__ == "__main__":
    main()
