# RQ1 Visualization Code

This document stores the plotting code for the formal RQ1 results. All heatmap code should be kept here so the visualization procedure is reproducible.

## Purpose

The main visualization compares perturbation effects before and after baseline correction.

Definitions:

```text
uncorrected_drift = 1 - perturbation_similarity
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

Interpretation:

```text
uncorrected_drift shows the raw semantic distance between original-prompt outputs and perturbed-prompt outputs.
noise_corrected_drift subtracts the ordinary same-prompt sampling noise baseline.
```

Input file:

```text
outputs/sbert_rq1_formal_perturbation_effects_by_item.csv
```

Generated CSV files:

```text
outputs/sbert_rq1_formal_uncorrected_perturbation_summary.csv
outputs/sbert_rq1_formal_uncorrected_heatmap_drift.csv
outputs/sbert_rq1_formal_corrected_heatmap_drift.csv
```

Generated figure files:

```text
figures/rq1_uncorrected_drift_heatmap.png
figures/rq1_noise_corrected_drift_heatmap.png
figures/rq1_uncorrected_vs_corrected_heatmaps.png
```

## Python Plotting Code

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path("/Users/wenfenglin/Desktop/Pioneer")
INPUT = ROOT / "outputs" / "sbert_rq1_formal_perturbation_effects_by_item.csv"

FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

UNCORRECTED_SUMMARY = (
    ROOT / "outputs" / "sbert_rq1_formal_uncorrected_perturbation_summary.csv"
)
UNCORRECTED_HEATMAP = (
    ROOT / "outputs" / "sbert_rq1_formal_uncorrected_heatmap_drift.csv"
)
CORRECTED_HEATMAP = (
    ROOT / "outputs" / "sbert_rq1_formal_corrected_heatmap_drift.csv"
)

UNCORRECTED_FIGURE = FIGURE_DIR / "rq1_uncorrected_drift_heatmap.png"
CORRECTED_FIGURE = FIGURE_DIR / "rq1_noise_corrected_drift_heatmap.png"
COMPARISON_FIGURE = FIGURE_DIR / "rq1_uncorrected_vs_corrected_heatmaps.png"

TASK_ORDER = [
    "code_generation",
    "factual_qa",
    "math_reasoning",
    "open_ended_writing",
]

PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def ordered_heatmap(summary: pd.DataFrame, value_column: str) -> pd.DataFrame:
    heatmap = summary.pivot(
        index="perturbation_type",
        columns="task_type",
        values=value_column,
    )
    return heatmap.loc[PERTURBATION_ORDER, TASK_ORDER]


def save_single_heatmap(
    heatmap: pd.DataFrame,
    output_path: Path,
    title: str,
    cmap: str,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean drift"},
    )
    plt.title(title, pad=14)
    plt.xlabel("Task type")
    plt.ylabel("Perturbation type")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = pd.read_csv(INPUT)
    df["perturbation_similarity"] = df["perturbation_similarity"].astype(float)
    df["noise_corrected_drift"] = df["noise_corrected_drift"].astype(float)
    df["uncorrected_drift"] = 1 - df["perturbation_similarity"]

    uncorrected_summary = (
        df.groupby(["task_type", "perturbation_type"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            mean_uncorrected_drift=("uncorrected_drift", "mean"),
            std_uncorrected_drift=("uncorrected_drift", "std"),
        )
        .round(6)
    )
    uncorrected_summary.to_csv(UNCORRECTED_SUMMARY, index=False)

    corrected_summary = (
        df.groupby(["task_type", "perturbation_type"], as_index=False)
        .agg(mean_noise_corrected_drift=("noise_corrected_drift", "mean"))
        .round(6)
    )

    uncorrected_heatmap = ordered_heatmap(
        uncorrected_summary,
        "mean_uncorrected_drift",
    )
    corrected_heatmap = ordered_heatmap(
        corrected_summary,
        "mean_noise_corrected_drift",
    )

    uncorrected_heatmap.to_csv(UNCORRECTED_HEATMAP)
    corrected_heatmap.to_csv(CORRECTED_HEATMAP)

    save_single_heatmap(
        uncorrected_heatmap,
        UNCORRECTED_FIGURE,
        "RQ1 Raw Perturbation Drift Before Baseline Correction",
        cmap="YlOrRd",
        vmin=0,
    )

    corrected_abs_max = max(
        abs(float(corrected_heatmap.min().min())),
        abs(float(corrected_heatmap.max().max())),
    )
    save_single_heatmap(
        corrected_heatmap,
        CORRECTED_FIGURE,
        "RQ1 Noise-Corrected Perturbation Drift After Baseline Correction",
        cmap="RdBu_r",
        center=0,
        vmin=-corrected_abs_max,
        vmax=corrected_abs_max,
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(
        uncorrected_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean raw drift"},
        ax=axes[0],
    )
    axes[0].set_title("Before Baseline Correction")
    axes[0].set_xlabel("Task type")
    axes[0].set_ylabel("Perturbation type")
    axes[0].tick_params(axis="x", rotation=25)

    sns.heatmap(
        corrected_heatmap,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-corrected_abs_max,
        vmax=corrected_abs_max,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean corrected drift"},
        ax=axes[1],
    )
    axes[1].set_title("After Baseline Correction")
    axes[1].set_xlabel("Task type")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=25)

    plt.suptitle("RQ1 Perturbation Drift Before vs After Baseline Correction", y=1.02)
    plt.tight_layout()
    plt.savefig(COMPARISON_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Wrote {UNCORRECTED_SUMMARY}")
    print(f"Wrote {UNCORRECTED_HEATMAP}")
    print(f"Wrote {CORRECTED_HEATMAP}")
    print(f"Wrote {UNCORRECTED_FIGURE}")
    print(f"Wrote {CORRECTED_FIGURE}")
    print(f"Wrote {COMPARISON_FIGURE}")


if __name__ == "__main__":
    main()
```
