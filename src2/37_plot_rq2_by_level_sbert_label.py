"""Plot RQ2 by-level dose response with S-BERT labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "rq2_results" / "rq2_figures"
PREFIX = "rq2_surface_noise_combined_balanced_50x5_l0_2_4_8_16"
INPUT = FIGURE_DIR / f"{PREFIX}_by_level.csv"
OUTPUT = FIGURE_DIR / f"{PREFIX}_by_level_sbert.png"


def annotate_n(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series, n_values: pd.Series) -> None:
    y_span = float(y_values.max() - y_values.min())
    offset = y_span * 0.06 if y_span > 0 else 0.01
    for x, y, n in zip(x_values, y_values, n_values):
        ax.text(float(x), float(y) + offset, f"n={int(n)}", ha="center", va="bottom", fontsize=9, color="#4d4d4d")


def plot() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing input CSV: {INPUT}")

    df = pd.read_csv(INPUT).sort_values("strength_edits")
    x = df["strength_edits"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True)
    fig.suptitle("RQ2 Surface Noise Combined: Three Tasks + LongFactQA", fontsize=18, y=0.985)

    panels = [
        (
            axes[0, 0],
            "mean_cross_similarity",
            "Mean S-BERT",
            "#2F80ED",
        ),
        (
            axes[0, 1],
            "mean_abs_repeated_pass_rate_change",
            "Abs correctness change",
            "#D1495B",
        ),
        (
            axes[1, 0],
            "mean_noise_corrected_drift",
            "S-BERT noise-corrected drift",
            "#3E7C8E",
        ),
        (
            axes[1, 1],
            "share_harmful_correctness_drop",
            "Harmful drop share",
            "#8F44AD",
        ),
    ]

    for ax, column, ylabel, color in panels:
        y = df[column]
        ax.plot(x, y, marker="o", linewidth=2.4, markersize=7, color=color)
        annotate_n(ax, x, y, df["n"])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(x=0.05, y=0.12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    for ax in axes[1, :]:
        ax.set_xlabel("Surface-noise strength: corrupted instruction words")

    for ax in axes.ravel():
        ax.set_xticks(x)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    plot()
