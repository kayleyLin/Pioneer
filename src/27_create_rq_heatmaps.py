"""Create candidate heatmaps for RQ1/RQ2 reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RQ2_SUMMARY = ROOT / "rq2" / "outputs" / "rq2_formal_available_drift_performance_summary.csv"
RQ2_PERFORMANCE = ROOT / "rq2" / "outputs" / "rq2_formal_available_performance_change_summary.csv"
RQ2_CORR = ROOT / "rq2" / "outputs" / "rq2_stats_correlations.csv"

TASK_ORDER = ["code_generation", "factual_qa", "math_reasoning"]
PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def ensure_figures_dir() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def pivot_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    pivot = df.pivot(index="perturbation_type", columns="task_type", values=value)
    return pivot.reindex(index=PERTURBATION_ORDER, columns=TASK_ORDER)


def save_heatmap(
    data: pd.DataFrame,
    path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "vlag",
    center: float | None = 0.0,
    fmt: str = ".3f",
) -> None:
    plt.figure(figsize=(8.2, 5.2))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Task type")
    ax.set_ylabel("Perturbation type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_figures_dir()

    summary = pd.read_csv(RQ2_SUMMARY)
    performance = pd.read_csv(RQ2_PERFORMANCE)
    correlations = pd.read_csv(RQ2_CORR)

    drift = pivot_table(summary, "mean_noise_corrected_drift")
    perf_change = pivot_table(summary, "mean_absolute_performance_change")
    pdr = pivot_table(performance, "mean_pdr")

    corr_task_pert = correlations[
        correlations["task_type"].notna()
        & correlations["perturbation_type"].notna()
    ].copy()
    pearson = pivot_table(corr_task_pert, "pearson_r")
    spearman = pivot_table(corr_task_pert, "spearman_rho")

    save_heatmap(
        drift,
        FIGURES / "rq1_rq2_noise_corrected_drift_heatmap.png",
        "RQ1/RQ2 Predictor: Mean Noise-Corrected Semantic Drift",
        "Mean noise-corrected drift",
        cmap="vlag",
        center=0.0,
    )
    save_heatmap(
        perf_change,
        FIGURES / "rq2_performance_change_heatmap.png",
        "RQ2 Outcome: Mean Absolute Performance Change",
        "Original performance - perturbed performance",
        cmap="rocket_r",
        center=None,
    )
    save_heatmap(
        pdr,
        FIGURES / "rq2_pdr_heatmap.png",
        "RQ2 Literature-Aligned Metric: Mean PDR",
        "Mean performance drop rate",
        cmap="vlag",
        center=0.0,
    )
    save_heatmap(
        pearson,
        FIGURES / "rq2_pearson_drift_performance_correlation_heatmap.png",
        "RQ2 Association: Pearson Correlation Between Drift And Performance Change",
        "Pearson r",
        cmap="vlag",
        center=0.0,
    )
    save_heatmap(
        spearman,
        FIGURES / "rq2_spearman_drift_performance_correlation_heatmap.png",
        "RQ2 Association: Spearman Correlation Between Drift And Performance Change",
        "Spearman rho",
        cmap="vlag",
        center=0.0,
    )

    # Combined contrast heatmap: performance change minus semantic drift after
    # standardizing each matrix. This is exploratory and helps identify cells
    # where performance loss is larger than semantic drift would suggest.
    drift_z = (drift - drift.stack().mean()) / drift.stack().std()
    perf_z = (perf_change - perf_change.stack().mean()) / perf_change.stack().std()
    contrast = perf_z - drift_z
    save_heatmap(
        contrast,
        FIGURES / "rq2_performance_vs_drift_contrast_heatmap.png",
        "RQ2 Exploratory Contrast: Performance Change Relative To Semantic Drift",
        "z(performance change) - z(drift)",
        cmap="vlag",
        center=0.0,
    )

    print("Wrote heatmaps to", FIGURES)
    for path in sorted(FIGURES.glob("*heatmap.png")):
        print(path)


if __name__ == "__main__":
    main()
