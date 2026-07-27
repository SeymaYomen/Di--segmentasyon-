from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {"Baseline": "#2563EB", "CLAHE": "#EA580C"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>"
        "text{font-family:Arial,sans-serif;fill:#172033}"
        ".title{font-size:20px;font-weight:700}"
        ".label{font-size:13px}.small{font-size:11px;fill:#526071}"
        ".grid{stroke:#D9E0E8;stroke-width:1}"
        "</style>",
        f'<text x="{width / 2}" y="30" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def save_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*lines, "</svg>"]), encoding="utf-8")


def performance_figure(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 980, 530
    left, top, bottom, plot_width = 75, 65, 420, 850
    metrics = [("dice", "Dice"), ("iou", "IoU")]
    groups = []
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        for metric, label in metrics:
            groups.append((dataset, metric, label))

    lines = svg_start(width, height, "İç ve dış test segmentasyon performansı")
    for tick in range(0, 101, 10):
        y = bottom - (bottom - top) * tick / 100
        lines += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick}</text>',
        ]

    group_width = plot_width / len(groups)
    bar_width = min(24, group_width * 0.28)
    for index, (dataset, metric, metric_label) in enumerate(groups):
        center = left + group_width * (index + 0.5)
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for method_index, method in enumerate(("Baseline", "CLAHE")):
            row = next(row for row in dataset_rows if row["method"] == method)
            value = float(row[metric]) * 100
            x = center + (method_index - 0.5) * (bar_width + 4)
            y = bottom - (bottom - top) * value / 100
            lines += [
                f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width}" '
                f'height="{bottom - y:.1f}" rx="2" fill="{COLORS[method]}"/>',
                f'<text x="{x:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="small">{value:.1f}</text>',
            ]
        short_dataset = dataset.replace(" exploratory", "")
        lines += [
            f'<text x="{center:.1f}" y="{bottom + 18}" text-anchor="middle" class="small">{escape(metric_label)}</text>',
            f'<text x="{center:.1f}" y="{bottom + 34}" text-anchor="middle" class="small">{escape(short_dataset)}</text>',
        ]

    lines += [
        f'<text x="20" y="{(top + bottom) / 2}" transform="rotate(-90 20 {(top + bottom) / 2})" '
        'text-anchor="middle" class="label">Skor (%)</text>',
        f'<rect x="365" y="478" width="14" height="14" fill="{COLORS["Baseline"]}"/>',
        '<text x="386" y="490" class="label">Baseline</text>',
        f'<rect x="490" y="478" width="14" height="14" fill="{COLORS["CLAHE"]}"/>',
        '<text x="511" y="490" class="label">CLAHE</text>',
    ]
    save_svg(output, lines)


def conformal_figure(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 700, 430
    left, top, bottom, plot_width = 75, 65, 350, 550
    lines = svg_start(width, height, "Conformal kapsama ve belirsizlik karşılaştırması")
    for tick in range(0, 101, 10):
        y = bottom - (bottom - top) * tick / 100
        lines += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick}</text>',
        ]
    categories = [
        ("empirical_pixel_coverage", "Ampirik kapsama"),
        ("ambiguous_or_empty_fraction", "Belirsiz/boş oranı"),
    ]
    for category_index, (column, label) in enumerate(categories):
        center = left + plot_width * (category_index + 0.5) / len(categories)
        for method_index, row in enumerate(rows):
            method = row["method"]
            value = float(row[column]) * 100
            x = center + (method_index - 0.5) * 52
            y = bottom - (bottom - top) * value / 100
            lines += [
                f'<rect x="{x - 20}" y="{y:.1f}" width="40" height="{bottom - y:.1f}" '
                f'rx="3" fill="{COLORS[method]}"/>',
                f'<text x="{x}" y="{y - 6:.1f}" text-anchor="middle" class="label">{value:.2f}</text>',
            ]
        lines.append(
            f'<text x="{center}" y="{bottom + 28}" text-anchor="middle" class="label">{escape(label)}</text>'
        )
    lines += [
        f'<rect x="225" y="392" width="14" height="14" fill="{COLORS["Baseline"]}"/>',
        '<text x="246" y="404" class="label">Baseline</text>',
        f'<rect x="360" y="392" width="14" height="14" fill="{COLORS["CLAHE"]}"/>',
        '<text x="381" y="404" class="label">CLAHE</text>',
    ]
    save_svg(output, lines)


def subgroup_figure(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 760, 450
    left, top, bottom, plot_width = 85, 65, 360, 600
    minimum, maximum = 0.88, 0.96
    lines = svg_start(width, height, "Yaş alt gruplarında Dice ve %95 bootstrap güven aralığı")
    for tick_index in range(5):
        value = minimum + (maximum - minimum) * tick_index / 4
        y = bottom - (bottom - top) * (value - minimum) / (maximum - minimum)
        lines += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.2f}</text>',
        ]
    groups = list(dict.fromkeys(row["age_group"] for row in rows))
    for group_index, group in enumerate(groups):
        center = left + plot_width * (group_index + 0.5) / len(groups)
        group_rows = [row for row in rows if row["age_group"] == group]
        for method_index, row in enumerate(group_rows):
            method = row["method"]
            value = float(row["dice"])
            low, high = float(row["ci95_low"]), float(row["ci95_high"])
            x = center + (method_index - 0.5) * 70
            y = bottom - (bottom - top) * (value - minimum) / (maximum - minimum)
            y_low = bottom - (bottom - top) * (low - minimum) / (maximum - minimum)
            y_high = bottom - (bottom - top) * (high - minimum) / (maximum - minimum)
            lines += [
                f'<line x1="{x}" y1="{y_low:.1f}" x2="{x}" y2="{y_high:.1f}" '
                f'stroke="{COLORS[method]}" stroke-width="3"/>',
                f'<line x1="{x - 8}" y1="{y_low:.1f}" x2="{x + 8}" y2="{y_low:.1f}" '
                f'stroke="{COLORS[method]}" stroke-width="2"/>',
                f'<line x1="{x - 8}" y1="{y_high:.1f}" x2="{x + 8}" y2="{y_high:.1f}" '
                f'stroke="{COLORS[method]}" stroke-width="2"/>',
                f'<circle cx="{x}" cy="{y:.1f}" r="7" fill="{COLORS[method]}"/>',
                f'<text x="{x}" y="{y - 12:.1f}" text-anchor="middle" class="small">{value:.3f}</text>',
            ]
        label = "Yetişkin (n=330)" if group == "adult" else "Çocuk (n=30)"
        lines.append(f'<text x="{center}" y="{bottom + 30}" text-anchor="middle" class="label">{label}</text>')
    lines += [
        f'<rect x="255" y="412" width="14" height="14" fill="{COLORS["Baseline"]}"/>',
        '<text x="276" y="424" class="label">Baseline</text>',
        f'<rect x="390" y="412" width="14" height="14" fill="{COLORS["CLAHE"]}"/>',
        '<text x="411" y="424" class="label">CLAHE</text>',
    ]
    save_svg(output, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toplu sonuçlardan yayımlanabilir SVG grafikler üretir.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/published"))
    args = parser.parse_args()
    results = args.results_dir
    performance_figure(read_csv(results / "final_metrics.csv"), results / "model_performance.svg")
    conformal_figure(read_csv(results / "conformal_comparison.csv"), results / "conformal_comparison.svg")
    subgroup_figure(read_csv(results / "age_subgroup_metrics.csv"), results / "age_subgroup_dice.svg")
    print(f"3 SVG grafik üretildi: {results.resolve()}")


if __name__ == "__main__":
    main()
