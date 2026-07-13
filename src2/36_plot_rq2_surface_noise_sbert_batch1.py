"""Recompute RQ2 surface-noise batch-1 similarity with Sentence-BERT and plot.

Input:
    rq2_results/rq2_surface_noise_combined_balanced_50x5_l0_2_4_8_16_batch1of5_generations.csv

Outputs:
    rq2_results/figure/rq2_surface_noise_combined_balanced_50x5_l0_2_4_8_16_batch1of5_sbert_by_case.csv
    rq2_results/figure/rq2_surface_noise_combined_balanced_50x5_l0_2_4_8_16_batch1of5_sbert_by_level.csv
    rq2_results/figure/rq2_surface_noise_combined_balanced_50x5_l0_2_4_8_16_batch1of5_sbert_similarity_correctness.png
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PREFIX = "rq2_surface_noise_combined_balanced_50x5_l0_2_4_8_16_batch1of5"
INPUT = ROOT / "rq2_results" / f"{PREFIX}_generations.csv"
FIGURE_DIR = ROOT / "rq2_results" / "figure"
BY_CASE = FIGURE_DIR / f"{PREFIX}_sbert_by_case.csv"
BY_LEVEL = FIGURE_DIR / f"{PREFIX}_sbert_by_level.csv"
FIGURE = FIGURE_DIR / f"{PREFIX}_sbert_similarity_correctness.png"

TASK_ORDER = ["factual_qa", "long_factual_qa", "math_reasoning", "code_generation"]
TASK_LABELS = {
    "factual_qa": "Factual QA",
    "long_factual_qa": "Long factual QA",
    "math_reasoning": "Math reasoning",
    "code_generation": "Code generation",
}
PALETTE = {
    "factual_qa": "#4C72B0",
    "long_factual_qa": "#8172B2",
    "math_reasoning": "#DD8452",
    "code_generation": "#55A868",
}


def mean_or_nan(values: list[float]) -> float:
    clean = [float(value) for value in values if not pd.isna(value)]
    return float(np.mean(clean)) if clean else np.nan


def pairwise_mean(matrix: np.ndarray, left: list[int], right: list[int]) -> float:
    if not left or not right:
        return np.nan
    values = [matrix[i, j] for i, j in product(left, right)]
    return float(np.mean(values))


def within_mean(matrix: np.ndarray, indexes: list[int]) -> float:
    if len(indexes) < 2:
        return np.nan
    values = [matrix[i, j] for i, j in combinations(indexes, 2)]
    return float(np.mean(values))


def paired_mean(matrix: np.ndarray, original: pd.DataFrame, perturbed: pd.DataFrame) -> float:
    values = []
    original_by_sample = {
        int(row.sample_idx): int(row.embedding_idx) for row in original.itertuples()
    }
    for row in perturbed.itertuples():
        sample_idx = int(row.sample_idx)
        if sample_idx in original_by_sample:
            values.append(matrix[original_by_sample[sample_idx], int(row.embedding_idx)])
    return float(np.mean(values)) if values else np.nan


def build_case_rows(df: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer(MODEL_NAME)
    outputs = df["output"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        outputs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarity_matrix = embeddings @ embeddings.T
    df = df.copy()
    df["embedding_idx"] = np.arange(len(df))

    case_rows: list[dict] = []
    for base_case_id, case_df in df.groupby("base_case_id", sort=True):
        original = case_df[
            (case_df["version"] == "original") & (case_df["strength_level"] == 0)
        ].copy()
        original_indexes = original["embedding_idx"].astype(int).tolist()
        baseline_similarity = within_mean(similarity_matrix, original_indexes)
        clean_mean = mean_or_nan(original["performance_score"].tolist())
        task = str(case_df["task"].iloc[0])
        dataset = str(case_df["dataset"].iloc[0])

        for strength_level, level_df in case_df[
            case_df["version"] == "perturbed"
        ].groupby("strength_level", sort=True):
            perturbed_indexes = level_df["embedding_idx"].astype(int).tolist()
            cross_similarity = pairwise_mean(
                similarity_matrix, original_indexes, perturbed_indexes
            )
            matched_similarity = paired_mean(similarity_matrix, original, level_df)
            perturbed_mean = mean_or_nan(level_df["performance_score"].tolist())
            pass_rate_drop = clean_mean - perturbed_mean
            abs_change = abs(pass_rate_drop)
            changed = int(abs_change > 1e-12)
            harmful = int(pass_rate_drop > 1e-12)

            case_rows.append(
                {
                    "base_case_id": base_case_id,
                    "task": task,
                    "dataset": dataset,
                    "perturbation_family": "surface_noise",
                    "strength_level": int(strength_level),
                    "strength_edits": int(level_df["strength_edits"].iloc[0]),
                    "n_original_outputs": len(original),
                    "n_perturbed_outputs": len(level_df),
                    "sbert_baseline_similarity": baseline_similarity,
                    "sbert_mean_cross_similarity": cross_similarity,
                    "sbert_mean_paired_similarity": matched_similarity,
                    "sbert_raw_perturbation_drift": 1 - cross_similarity,
                    "sbert_noise_corrected_drift": baseline_similarity
                    - cross_similarity,
                    "clean_mean_correctness": clean_mean,
                    "perturbed_mean_correctness": perturbed_mean,
                    "repeated_pass_rate_drop": pass_rate_drop,
                    "abs_repeated_pass_rate_change": abs_change,
                    "harmful_correctness_drop": harmful,
                    "correctness_changed": changed,
                    "similarity_metric": MODEL_NAME,
                }
            )

    return pd.DataFrame(case_rows)


def summarize_by_level(case_df: pd.DataFrame) -> pd.DataFrame:
    grouped = case_df.groupby(["strength_level", "strength_edits"], as_index=False)
    summary = grouped.agg(
        n=("base_case_id", "count"),
        mean_sbert_cross_similarity=("sbert_mean_cross_similarity", "mean"),
        sd_sbert_cross_similarity=("sbert_mean_cross_similarity", "std"),
        mean_sbert_paired_similarity=("sbert_mean_paired_similarity", "mean"),
        mean_sbert_raw_perturbation_drift=("sbert_raw_perturbation_drift", "mean"),
        mean_sbert_noise_corrected_drift=("sbert_noise_corrected_drift", "mean"),
        mean_clean_correctness=("clean_mean_correctness", "mean"),
        mean_perturbed_correctness=("perturbed_mean_correctness", "mean"),
        mean_abs_repeated_pass_rate_change=("abs_repeated_pass_rate_change", "mean"),
        mean_repeated_pass_rate_drop=("repeated_pass_rate_drop", "mean"),
        share_harmful_correctness_drop=("harmful_correctness_drop", "mean"),
        share_correctness_changed=("correctness_changed", "mean"),
    )
    summary["se_sbert_cross_similarity"] = (
        summary["sd_sbert_cross_similarity"] / np.sqrt(summary["n"])
    )
    summary["ci95_sbert_cross_similarity"] = (
        1.96 * summary["se_sbert_cross_similarity"]
    )
    summary["similarity_metric"] = MODEL_NAME
    return summary


def plot(case_df: pd.DataFrame, level_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        gridspec_kw={"width_ratios": [1.05, 1.25]},
        constrained_layout=True,
    )

    ax = axes[0]
    x = level_df["strength_edits"].to_numpy()
    y = level_df["mean_sbert_cross_similarity"].to_numpy()
    yerr = level_df["ci95_sbert_cross_similarity"].fillna(0).to_numpy()
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color="#3B5B92",
        marker="o",
        linewidth=2.1,
        capsize=3,
        label="SBERT cross similarity",
    )
    ax.set_title("Overall surface-noise dose response")
    ax.set_xlabel("Surface-noise edits")
    ax.set_ylabel("Mean SBERT cross similarity")
    ax.set_xticks(x)
    ax.set_ylim(max(0.0, min(y - yerr) - 0.02), min(1.01, max(y + yerr) + 0.02))

    ax2 = ax.twinx()
    ax2.plot(
        x,
        level_df["mean_abs_repeated_pass_rate_change"],
        color="#B45A4A",
        marker="s",
        linewidth=1.8,
        label="Abs. performance change",
    )
    ax2.plot(
        x,
        level_df["share_harmful_correctness_drop"],
        color="#767676",
        marker="^",
        linewidth=1.6,
        linestyle="--",
        label="Share harmful drop",
    )
    ax2.set_ylabel("Correctness / performance statistic")
    ax2.set_ylim(0, max(0.35, level_df["share_harmful_correctness_drop"].max() + 0.08))

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left", frameon=True)

    ax = axes[1]
    task_df = (
        case_df.groupby(["task", "strength_edits"], as_index=False)
        .agg(
            mean_sbert_cross_similarity=("sbert_mean_cross_similarity", "mean"),
            mean_abs_change=("abs_repeated_pass_rate_change", "mean"),
        )
        .sort_values(["task", "strength_edits"])
    )
    for task in TASK_ORDER:
        sub = task_df[task_df["task"] == task]
        if sub.empty:
            continue
        ax.plot(
            sub["strength_edits"],
            sub["mean_sbert_cross_similarity"],
            marker="o",
            linewidth=1.9,
            color=PALETTE[task],
            label=TASK_LABELS[task],
        )
    ax.set_title("SBERT similarity by task")
    ax.set_xlabel("Surface-noise edits")
    ax.set_ylabel("Mean SBERT cross similarity")
    ax.set_xticks(x)
    ax.set_ylim(
        max(0.0, task_df["mean_sbert_cross_similarity"].min() - 0.03),
        min(1.01, task_df["mean_sbert_cross_similarity"].max() + 0.02),
    )
    ax.legend(loc="lower left", frameon=True)

    fig.suptitle(
        "Surface noise strength vs. SBERT similarity and correctness, batch 1 of 5",
        fontsize=12,
        y=1.03,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing input file: {INPUT}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    case_df = build_case_rows(df)
    level_df = summarize_by_level(case_df)

    case_df.to_csv(BY_CASE, index=False)
    level_df.to_csv(BY_LEVEL, index=False)
    plot(case_df, level_df)

    print(f"Wrote {BY_CASE}")
    print(f"Wrote {BY_LEVEL}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
