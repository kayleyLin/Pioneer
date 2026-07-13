"""Build codex_prompt Figures 1-3 for Llama 3.3 70B n=50 four-task outputs.

This script recomputes the SBERT n=50 baseline and perturbation tables from
llama/llama33_70b_instruct_turbo_150case/outputs, then renders Figures 1-3 into
llama/llama33_70b_instruct_turbo_150case/figures.

Figure 4 must use the real correctness/performance pipeline in
src/build_llama33_real_fig4.py, not the older draft placeholder logic.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "llama" / "llama33_70b_instruct_turbo_150case"
OUTPUTS = BASE / "outputs"
FIGURES = BASE / "figures"

ORIGINAL_FOUR_TASK = OUTPUTS / "rq1_llama33_70b_original_generations_n50_four_task.csv"
PERTURBED_FOUR_TASK = OUTPUTS / "rq1_llama33_70b_perturbed_generations_n50_four_task.csv"

BASELINE_BY_ITEM = OUTPUTS / "sbert_rq1_n50_baseline_by_item.csv"
BASELINE_BY_TASK = OUTPUTS / "sbert_rq1_n50_baseline_by_task.csv"
TESTS = OUTPUTS / "rq1_n50_baseline_significance_tests.csv"
TUKEY = OUTPUTS / "rq1_n50_baseline_tukey.csv"
PERTURBATION_BY_ITEM = OUTPUTS / "sbert_rq1_n50_perturbation_effects_by_item.csv"
PERTURBATION_SUMMARY = OUTPUTS / "sbert_rq1_n50_perturbation_summary.csv"
CORRECTED_HEATMAP = OUTPUTS / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv"
UNCORRECTED_SUMMARY = OUTPUTS / "sbert_rq1_n50_uncorrected_perturbation_summary.csv"
UNCORRECTED_HEATMAP = OUTPUTS / "sbert_rq1_n50_uncorrected_heatmap_drift.csv"
FIG4_DRAFT_DATA = OUTPUTS / "fig4_similarity_correctness_draft_data.csv"

TASK_ORDER = [
    "factual_qa",
    "math_reasoning",
    "code_generation",
    "open_ended_writing",
]
TASK_LABELS = {
    "factual_qa": "Factual QA",
    "math_reasoning": "Math reasoning",
    "code_generation": "Code generation",
    "open_ended_writing": "Open-ended writing",
}
TASK_COLORS = {
    "factual_qa": "#4C72B0",
    "math_reasoning": "#DD8452",
    "code_generation": "#55A868",
    "open_ended_writing": "#C44E52",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_inputs() -> None:
    missing = [
        str(path)
        for path in [ORIGINAL_FOUR_TASK, PERTURBED_FOUR_TASK]
        if not path.exists()
    ]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))


def recompute_sbert_tables() -> None:
    baseline = load_module(
        "llama_n50_baseline",
        ROOT / "llama" / "src" / "31_analyze_rq1_n50_baseline_sbert_llama.py",
    )
    baseline.GENERATION_FILES = [ORIGINAL_FOUR_TASK]
    baseline.BY_ITEM = BASELINE_BY_ITEM
    baseline.BY_TASK = BASELINE_BY_TASK
    baseline.TESTS = TESTS
    baseline.TUKEY = TUKEY
    baseline.main()

    perturbation = load_module(
        "llama_n50_perturbation",
        ROOT / "llama" / "src" / "32_analyze_rq1_n50_perturbations_sbert_llama.py",
    )
    perturbation.ORIGINAL_FILES = [ORIGINAL_FOUR_TASK]
    perturbation.PERTURBED_FILES = [PERTURBED_FOUR_TASK]
    perturbation.BY_ITEM = PERTURBATION_BY_ITEM
    perturbation.SUMMARY = PERTURBATION_SUMMARY
    perturbation.HEATMAP = CORRECTED_HEATMAP
    perturbation.UNCORRECTED_SUMMARY = UNCORRECTED_SUMMARY
    perturbation.UNCORRECTED_HEATMAP = UNCORRECTED_HEATMAP
    perturbation.main()


def render_figures() -> list[Path]:
    figures = load_module(
        "llama_codex_prompt_figures",
        ROOT / "llama" / "src" / "35_create_codex_prompt_figures_llama.py",
    )
    figures.OUTPUTS = OUTPUTS
    figures.FIGURES = FIGURES
    figures.BASELINE_BY_ITEM = BASELINE_BY_ITEM
    figures.BASELINE_BY_TASK = BASELINE_BY_TASK
    figures.TUKEY = TUKEY
    figures.UNCORRECTED_HEATMAP = UNCORRECTED_HEATMAP
    figures.CORRECTED_HEATMAP = CORRECTED_HEATMAP
    figures.PERTURBATION_BY_ITEM = PERTURBATION_BY_ITEM
    figures.FIG4_DRAFT_DATA = FIG4_DRAFT_DATA

    figures.configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    return [
        figures.create_fig1(),
        figures.create_fig2(),
        figures.create_fig3(),
    ]


def expected_retention_probability(similarity: float, task_type: str) -> float:
    """Deterministic draft curve used only when true correctness labels are absent."""
    task_adjustment = {
        "factual_qa": 0.04,
        "math_reasoning": -0.08,
        "code_generation": -0.03,
    }[task_type]
    midpoint = 0.80
    slope = 13.0
    probability = 1 / (1 + math.exp(-slope * (similarity - midpoint)))
    probability = 0.04 + 0.92 * probability + task_adjustment
    return min(0.98, max(0.02, probability))


def create_fig4_draft() -> Path:
    source = pd.read_csv(PERTURBATION_BY_ITEM)
    source = source[
        source["task_type"].isin(["factual_qa", "math_reasoning", "code_generation"])
    ].copy()
    source["similarity"] = source["perturbation_similarity"].astype(float)
    source["expected_retention_rate"] = source.apply(
        lambda row: expected_retention_probability(row["similarity"], row["task_type"]),
        axis=1,
    )

    bins = [-np.inf, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001]
    labels_low_to_high = [
        "<0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]
    labels = list(reversed(labels_low_to_high))
    source["similarity_bin"] = pd.cut(
        source["similarity"],
        bins=bins,
        labels=labels_low_to_high,
        include_lowest=True,
        right=False,
    )

    grouped = (
        source.groupby(["task_type", "similarity_bin"], observed=False)
        .agg(
            retention_rate=("expected_retention_rate", "mean"),
            n=("expected_retention_rate", "size"),
            mean_similarity=("similarity", "mean"),
        )
        .reset_index()
    )
    overall = (
        source.groupby("similarity_bin", observed=False)
        .agg(
            retention_rate=("expected_retention_rate", "mean"),
            n=("expected_retention_rate", "size"),
            mean_similarity=("similarity", "mean"),
        )
        .reset_index()
    )
    grouped["similarity_bin"] = pd.Categorical(
        grouped["similarity_bin"], categories=labels, ordered=True
    )
    overall["similarity_bin"] = pd.Categorical(
        overall["similarity_bin"], categories=labels, ordered=True
    )
    grouped = grouped.sort_values(["task_type", "similarity_bin"])
    overall = overall.sort_values("similarity_bin")

    source[
        [
            "item_id",
            "task_type",
            "perturbation_type",
            "similarity",
            "similarity_bin",
            "expected_retention_rate",
            "similarity_metric",
        ]
    ].to_csv(FIG4_DRAFT_DATA, index=False)
    overall.to_csv(OUTPUTS / "fig4_similarity_correctness_draft_bins_overall.csv", index=False)
    grouped.to_csv(OUTPUTS / "fig4_similarity_correctness_draft_bins_by_task.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(labels))
    overall_lookup = overall.set_index("similarity_bin")
    bar_values = [float(overall_lookup.loc[label, "retention_rate"]) for label in labels]
    bar_counts = [int(overall_lookup.loc[label, "n"]) for label in labels]
    bars = ax.bar(
        x,
        bar_values,
        color="#B8B8B8",
        edgecolor="#6F6F6F",
        linewidth=0.8,
        width=0.68,
        label="Overall expected rate",
        alpha=0.72,
    )
    for rect, value, count in zip(bars, bar_values, bar_counts):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            value + 0.025,
            f"{value:.0%}\nn={count}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#333333",
        )

    for task in ["factual_qa", "math_reasoning", "code_generation"]:
        task_rows = grouped[grouped["task_type"] == task].set_index("similarity_bin")
        y_values = []
        for label in labels:
            n = int(task_rows.loc[label, "n"])
            y_values.append(float(task_rows.loc[label, "retention_rate"]) if n > 0 else np.nan)
        ax.plot(
            x,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.set_xlabel("Output similarity bin (high to low)")
    ax.set_ylabel("Expected correctness retention rate")
    ax.set_title("Similarity Bins vs Correctness Retention (Draft Placeholder)")
    ax.grid(axis="y", linestyle="--", color="#D5D5D5", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.text(
        0.5,
        0.02,
        "Note: true correctness labels are unavailable; this draft uses deterministic expected retention from similarity, not sampled correctness.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#444444",
    )
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    path = FIGURES / "fig4_similarity_correctness_draft.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def validate_outputs(paths: list[Path]) -> None:
    baseline = pd.read_csv(BASELINE_BY_ITEM)
    perturbation = pd.read_csv(PERTURBATION_BY_ITEM)
    corrected = pd.read_csv(CORRECTED_HEATMAP)
    uncorrected = pd.read_csv(UNCORRECTED_HEATMAP)

    task_counts = baseline.groupby("task_type")["item_id"].nunique().to_dict()
    if task_counts != {task: 50 for task in TASK_ORDER}:
        raise SystemExit(f"Unexpected baseline item counts: {task_counts}")
    if len(baseline) != 200:
        raise SystemExit(f"Expected 200 baseline rows, got {len(baseline)}")
    if len(perturbation) != 1000:
        raise SystemExit(f"Expected 1000 perturbation rows, got {len(perturbation)}")
    if corrected.shape != (5, 5) or uncorrected.shape != (5, 5):
        raise SystemExit(
            "Expected 5-row heatmaps with perturbation_type plus four task columns"
        )

    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty figure: {path}")
        with Image.open(path) as image:
            width, height = image.size
        if width < 1000 or height < 900:
            raise SystemExit(f"Figure resolution looks too small: {path} {width}x{height}")

    print("Validation passed")
    print(f"Baseline task counts: {task_counts}")
    print(f"Perturbation item-perturbation rows: {len(perturbation)}")


def main() -> None:
    require_inputs()
    recompute_sbert_tables()
    paths = render_figures()
    validate_outputs(paths)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
