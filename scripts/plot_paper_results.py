"""Regenerate provider-neutral figures from the public aggregate CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

COLORS = {
    "Direct": "#8C8C8C",
    "CoT": "#333333",
    "Grounded-CoSQ": "#0072B2",
    "Critical-CoSQ": "#009E73",
    "Adaptive-CoSQ": "#D55E00",
}
MARKERS = {
    "Direct": "x",
    "CoT": "s",
    "Grounded-CoSQ": "o",
    "Critical-CoSQ": "^",
    "Adaptive-CoSQ": "D",
}
VARIANT_LABELS = {
    "grounded": "Grounded-CoSQ",
    "critical": "Critical-CoSQ",
    "adaptive": "Adaptive-CoSQ",
}
MODEL_LABELS = {
    "llama3-8b": "Llama 3 8B",
    "llama3-70b": "Llama 3 70B",
    "llama4_scout-17b": "Llama 4 Scout 17B",
    "gemma3_12b_it": "Gemma 3 12B",
    "gemma4_31b_it": "Gemma 4 31B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "mistral-7b": "Mistral 7B",
    "gpt5_5": "GPT-5.5",
    "claude5_sonnet": "Claude 5 Sonnet",
    "deepseek-flash": "DeepSeek Flash",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis: Axes) -> None:
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))


def save_figure(fig: Figure, output: Path, stem: str, formats: Iterable[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
            )
    plt.close(fig)


def plot_risk_coverage(output: Path, formats: Iterable[str]) -> None:
    rows = read_csv(RESULTS / "truthfulqa_condition_summary.csv")
    curves: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in VARIANT_LABELS
    }
    pattern = re.compile(r"^(grounded|critical|adaptive)_mc_mean(\d{3})$")
    cot = next(row for row in rows if row["condition"] == "cot_mc")

    for row in rows:
        match = pattern.match(row["condition"])
        if match:
            curves[match.group(1)].append(
                (
                    int(match.group(2)) / 100,
                    float(row["coverage_mean"]),
                    float(row["HR_mean"]),
                )
            )

    fig, axis = plt.subplots(figsize=(7.1, 4.5))
    for variant, points in curves.items():
        points.sort()
        label = VARIANT_LABELS[variant]
        axis.plot(
            [point[1] for point in points],
            [point[2] for point in points],
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=1.8,
            markersize=6,
            label=label,
        )
        for threshold, coverage, risk in points:
            if threshold in {0.50, 0.90}:
                axis.annotate(
                    f"{threshold:.2f}",
                    (coverage, risk),
                    xytext=(4, 5 if threshold == 0.50 else -12),
                    textcoords="offset points",
                    fontsize=8,
                    color=COLORS[label],
                )

    axis.scatter(
        float(cot["coverage_mean"]),
        float(cot["HR_mean"]),
        color=COLORS["CoT"],
        marker=MARKERS["CoT"],
        s=48,
        label="CoT",
        zorder=5,
    )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Unconditional wrong-commitment rate (HR)")
    axis.set_title("TruthfulQA-MC1 risk-coverage operating points")
    axis.set_xlim(0.84, 1.01)
    axis.set_ylim(0.084, 0.134)
    style_axis(axis)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    axis.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output, "truthfulqa_risk_coverage", formats)


def plot_truthfulqa_model_hr(output: Path, formats: Iterable[str]) -> None:
    rows = read_csv(RESULTS / "truthfulqa_model_conditions.csv")
    by_model: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["condition"] in {"cot_mc", "grounded_mc_mean090"}:
            by_model.setdefault(row["model_id"], {})[row["condition"]] = float(row["HR"])

    ordered = sorted(
        by_model,
        key=lambda model: by_model[model]["cot_mc"]
        - by_model[model]["grounded_mc_mean090"],
    )
    y_positions = list(range(len(ordered)))

    fig, axis = plt.subplots(figsize=(7.1, 5.5))
    for y, model in zip(y_positions, ordered, strict=True):
        cot = by_model[model]["cot_mc"]
        grounded = by_model[model]["grounded_mc_mean090"]
        axis.plot([grounded, cot], [y, y], color="#C8C8C8", linewidth=2, zorder=1)
    axis.scatter(
        [by_model[model]["cot_mc"] for model in ordered],
        y_positions,
        color=COLORS["CoT"],
        marker=MARKERS["CoT"],
        s=42,
        label="CoT",
        zorder=3,
    )
    axis.scatter(
        [by_model[model]["grounded_mc_mean090"] for model in ordered],
        y_positions,
        color=COLORS["Grounded-CoSQ"],
        marker=MARKERS["Grounded-CoSQ"],
        s=44,
        label="Grounded-CoSQ (0.90)",
        zorder=3,
    )
    axis.set_yticks(y_positions, [MODEL_LABELS.get(model, model) for model in ordered])
    axis.set_xlabel("Unconditional wrong-commitment rate (HR)")
    axis.set_title("TruthfulQA-MC1 model-level risk reduction")
    axis.set_xlim(left=0.0)
    style_axis(axis)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, output, "truthfulqa_hr_by_model", formats)


def plot_nq_panel(output: Path, formats: Iterable[str]) -> None:
    rows = read_csv(RESULTS / "nq_model_conditions.csv")
    order = ["Direct", "CoT", "Grounded-CoSQ", "Critical-CoSQ", "Adaptive-CoSQ"]
    models = list(dict.fromkeys(row["model"] for row in rows))
    lookup = {(row["model"], row["label"]): row for row in rows}

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 8.2), sharey=True)
    metrics = [
        ("AA", "Answered accuracy (AA)"),
        ("coverage", "Coverage"),
        ("HR", "Unconditional wrong-commitment rate (HR)"),
    ]
    offsets = [-0.24, -0.12, 0.0, 0.12, 0.24]
    y_base = list(range(len(models)))

    for axis, (field, label) in zip(axes, metrics, strict=True):
        for condition, offset in zip(order, offsets, strict=True):
            values = [float(lookup[(model, condition)][field]) for model in models]
            axis.scatter(
                values,
                [position + offset for position in y_base],
                color=COLORS[condition],
                marker=MARKERS[condition],
                s=31,
                label=condition,
            )
        axis.set_xlabel(label)
        axis.set_xlim(0.0, 1.02)
        axis.set_yticks(y_base, models)
        style_axis(axis)

    axes[0].set_title("NQ-Short model-level generalization panel")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save_figure(fig, output, "nq_model_panel", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("svg", "pdf", "png"),
        help="output format; repeat for multiple formats (default: svg and pdf)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formats = args.formats or ["svg", "pdf"]
    plot_risk_coverage(args.out, formats)
    plot_truthfulqa_model_hr(args.out, formats)
    plot_nq_panel(args.out, formats)
    print(f"Wrote paper figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
