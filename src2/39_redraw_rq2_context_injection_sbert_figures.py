"""Redraw selected RQ2 context-injection figures with S-BERT labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "rq2_results" / "rq2_figures"
OUT_DIR = ROOT / "fig"
PREFIX = "rq2_context_injection_combined_balanced_50x5_l0_2_4_8_16"
METRICS_GLOB = f"{PREFIX}_batch*of5_metrics.csv"

TASK_ORDER = ["factual_qa", "math_reasoning", "code_generation", "long_factual_qa"]
TASK_LABELS = {
    "factual_qa": "Factual QA",
    "math_reasoning": "Math",
    "code_generation": "Code",
    "long_factual_qa": "LongFactQA",
}
TASK_COLORS = {
    "factual_qa": "#4C8FEA",
    "math_reasoning": "#F4A259",
    "code_generation": "#2FB36D",
    "long_factual_qa": "#A65BE8",
}
STRENGTH_MARKERS = {2: "o", 4: "s", 8: "^", 16: "D"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def finish(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def annotate_n(ax: plt.Axes, x_values, y_values, n_values) -> None:
    y = pd.Series(y_values, dtype=float)
    y_span = float(y.max() - y.min())
    offset = y_span * 0.055 if y_span > 0 else 0.01
    for x, value, n in zip(x_values, y_values, n_values):
        ax.text(float(x), float(value) + offset, f"n={int(n)}", ha="center", va="bottom", fontsize=10, color="#4d4d4d")


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def add_linear_trend(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    if len(x) < 2 or x.nunique() < 2:
        return
    coeff = np.polyfit(x.astype(float), y.astype(float), deg=1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, coeff[0] * xs + coeff[1], color="#3f3f3f", linewidth=2.2, label="Linear trend")


def load_metrics() -> pd.DataFrame:
    files = sorted((ROOT / "rq2_results").glob(METRICS_GLOB))
    if not files:
        raise SystemExit(f"No metrics files matched {METRICS_GLOB}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def plot_overall_dose_response() -> None:
    df = pd.read_csv(SOURCE_DIR / f"{PREFIX}_by_level.csv").sort_values("strength_edits")
    x = df["strength_edits"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True)
    fig.suptitle("RQ2 Context Injection Combined: Three Tasks + LongFactQA", fontsize=18, y=0.985)
    panels = [
        (axes[0, 0], "mean_cross_similarity", "Mean S-BERT", "#2F80ED"),
        (axes[0, 1], "mean_abs_repeated_pass_rate_change", "Abs correctness change", "#D1495B"),
        (axes[1, 0], "mean_noise_corrected_drift", "S-BERT noise-corrected drift", "#3E7C8E"),
        (axes[1, 1], "share_harmful_correctness_drop", "Harmful drop share", "#8F44AD"),
    ]
    for ax, column, ylabel, color in panels:
        y = df[column]
        ax.plot(x, y, marker="o", linewidth=2.4, markersize=7, color=color)
        annotate_n(ax, x, y, df["n"])
        ax.set_ylabel(ylabel)
        ax.margins(x=0.05, y=0.12)
        finish(ax)
    for ax in axes[1, :]:
        ax.set_xlabel("Context-injection strength: inserted context units")
    for ax in axes.ravel():
        ax.set_xticks(x)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, f"{PREFIX}_overall_dose_response_sbert.png")


def plot_line_by_task(column: str, ylabel: str, title: str, out_name: str) -> None:
    df = pd.read_csv(SOURCE_DIR / f"{PREFIX}_by_task_level.csv")
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for task in TASK_ORDER:
        sub = df[df["task"] == task].sort_values("strength_edits")
        ax.plot(
            sub["strength_edits"],
            sub[column],
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
    ax.set_title(title)
    ax.set_xlabel("Context-injection strength: inserted context units")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["strength_edits"].unique()))
    ax.legend(frameon=True)
    finish(ax)
    save(fig, out_name)


def scatter_by_task(metrics: pd.DataFrame, x_col: str, x_label: str, out_name: str) -> None:
    df = metrics[metrics["strength_level"] > 0].copy()
    fig, ax = plt.subplots(figsize=(11.4, 7.2))
    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        ax.scatter(
            sub[x_col],
            sub["abs_repeated_pass_rate_change"],
            s=70,
            alpha=0.78,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
            edgecolor="white",
            linewidth=0.5,
        )
    add_linear_trend(ax, df[x_col], df["abs_repeated_pass_rate_change"])
    ax.set_title("Context Injection Combined: Case-Level Nonzero Strengths")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Absolute repeated pass-rate change")
    ax.legend(frameon=False, ncol=2)
    finish(ax)
    save(fig, out_name)


def plot_code_direction(metrics: pd.DataFrame) -> None:
    df = metrics[(metrics["task"] == "code_generation") & (metrics["strength_level"] > 0)].copy()
    df["direction"] = np.select(
        [df["repeated_pass_rate_drop"] > 1e-12, df["repeated_pass_rate_drop"] < -1e-12],
        ["Harmful drop", "Improved under perturbation"],
        default="Unchanged",
    )
    colors = {
        "Harmful drop": "#D95A6A",
        "Improved under perturbation": "#3AAFA9",
        "Unchanged": "#9AA5AE",
    }
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    for direction, sub_direction in df.groupby("direction"):
        for edits, sub in sub_direction.groupby("strength_edits"):
            ax.scatter(
                sub["mean_cross_similarity"],
                sub["abs_repeated_pass_rate_change"],
                s=70,
                alpha=0.78,
                marker=STRENGTH_MARKERS.get(int(edits), "o"),
                color=colors[direction],
                edgecolor="white",
                linewidth=0.5,
                label=direction,
            )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    add_linear_trend(ax, df["mean_cross_similarity"], df["abs_repeated_pass_rate_change"])
    trend_handles, trend_labels = ax.get_legend_handles_labels()
    dedup.update(dict(zip(trend_labels, trend_handles)))
    rho = df["mean_cross_similarity"].corr(df["abs_repeated_pass_rate_change"], method="spearman")
    ax.text(
        0.02,
        0.96,
        f"Spearman rho = {rho:.4f}\nn = {len(df)} nonzero code rows",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.set_title("Code Generation: S-BERT vs Correctness Change under Context Injection")
    ax.set_xlabel("Mean S-BERT")
    ax.set_ylabel("Absolute repeated pass-rate change")
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="upper right")
    finish(ax)
    save(fig, f"{PREFIX}_code_only_similarity_vs_correctness_direction_sbert.png")


def plot_code_bin_trend() -> None:
    df = pd.read_csv(SOURCE_DIR / f"{PREFIX}_code_only_similarity_bins.csv").sort_values("mean_similarity", ascending=False)
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.errorbar(
        df["mean_similarity"],
        df["mean_abs_change"],
        yerr=df["se"],
        marker="o",
        markersize=8,
        linewidth=2.4,
        capsize=5,
        color="#3E7C8E",
    )
    annotate_n(ax, df["mean_similarity"], df["mean_abs_change"], df["n"])
    ax.set_title("Code Generation: Correctness Movement by S-BERT Bin")
    ax.set_xlabel("Mean S-BERT bin average (lower S-BERT to the right)")
    ax.set_ylabel("Mean absolute repeated pass-rate change")
    ax.invert_xaxis()
    finish(ax)
    save(fig, f"{PREFIX}_code_only_similarity_bin_trend_sbert.png")


def main() -> None:
    setup_style()
    metrics = load_metrics()
    plot_overall_dose_response()
    plot_line_by_task(
        "mean_abs_repeated_pass_rate_change",
        "Absolute repeated pass-rate change",
        "Context Injection Combined: Correctness Change by Task",
        f"{PREFIX}_correctness_change_by_task_sbert.png",
    )
    plot_line_by_task(
        "mean_cross_similarity",
        "Mean S-BERT",
        "Context Injection Combined: S-BERT by Task",
        f"{PREFIX}_similarity_by_task_sbert.png",
    )
    scatter_by_task(metrics, "mean_cross_similarity", "Mean S-BERT", f"{PREFIX}_similarity_vs_correctness_scatter_sbert.png")
    scatter_by_task(metrics, "noise_corrected_drift", "S-BERT noise-corrected drift", f"{PREFIX}_corrected_drift_vs_correctness_scatter_sbert.png")
    plot_code_direction(metrics)
    plot_code_bin_trend()


if __name__ == "__main__":
    main()
