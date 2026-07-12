"""Create Codex prompt paper figures 1-3 from Llama RQ1 n=50 results.

Outputs:
    llama/figures/fig1_noise_baseline.png
    llama/figures/fig2_tukey_forest.png
    llama/figures/fig3_drift_heatmap.png
    llama/figures/fig4_similarity_correctness_draft.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / "llama"
OUTPUTS = LLAMA / "outputs"
FIGURES = LLAMA / "figures"

BASELINE_BY_ITEM = OUTPUTS / "sbert_rq1_n50_baseline_by_item.csv"
BASELINE_BY_TASK = OUTPUTS / "sbert_rq1_n50_baseline_by_task.csv"
TUKEY = OUTPUTS / "rq1_n50_baseline_tukey.csv"
UNCORRECTED_HEATMAP = OUTPUTS / "sbert_rq1_n50_uncorrected_heatmap_drift.csv"
CORRECTED_HEATMAP = OUTPUTS / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv"
PERTURBATION_BY_ITEM = OUTPUTS / "sbert_rq1_n50_perturbation_effects_by_item.csv"
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

PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]
PERTURBATION_LABELS = {
    "paraphrasing": "Paraphrasing",
    "reordering": "Reordering",
    "formatting_changes": "Formatting changes",
    "context_injection": "Context injection",
    "surface_noise": "Surface noise",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))


def load_heatmap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("perturbation_type")
    df = df.reindex(index=PERTURBATION_ORDER, columns=TASK_ORDER).astype(float)
    df.index = [PERTURBATION_LABELS[item] for item in df.index]
    df.columns = [TASK_LABELS[item] for item in df.columns]
    return df.T


def annotate_heatmap(ax: plt.Axes, data: pd.DataFrame, cmap, norm) -> None:
    for y, row_name in enumerate(data.index):
        for x, col_name in enumerate(data.columns):
            value = float(data.loc[row_name, col_name])
            r, g, b, _ = cmap(norm(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "white" if luminance < 0.48 else "#222222"
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
            )


def compact_letters(tasks: list[str], tukey: pd.DataFrame) -> dict[str, str]:
    significant = {(task_a, task_b): False for task_a in tasks for task_b in tasks}
    for _, row in tukey.iterrows():
        a = str(row["group1"])
        b = str(row["group2"])
        reject = str(row["reject_alpha_0_05"]).lower() == "true"
        significant[(a, b)] = reject
        significant[(b, a)] = reject

    letters: list[list[str]] = []
    assigned: dict[str, str] = {task: "" for task in tasks}
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for task in tasks:
        placed = False
        for index, members in enumerate(letters):
            if all(not significant[(task, member)] for member in members):
                members.append(task)
                assigned[task] += alphabet[index]
                placed = True
        if not placed:
            letters.append([task])
            assigned[task] += alphabet[len(letters) - 1]
    return assigned


def create_fig1() -> Path:
    df = pd.read_csv(BASELINE_BY_ITEM)
    df["task_label"] = pd.Categorical(
        df["task_type"].map(TASK_LABELS),
        categories=[TASK_LABELS[item] for item in TASK_ORDER],
        ordered=True,
    )
    df["noise_drift"] = df["sampling_noise_drift"].astype(float)
    palette = [TASK_COLORS[item] for item in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    sns.violinplot(
        data=df,
        x="task_label",
        y="noise_drift",
        hue="task_label",
        order=[TASK_LABELS[item] for item in TASK_ORDER],
        hue_order=[TASK_LABELS[item] for item in TASK_ORDER],
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        saturation=0.85,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="task_label",
        y="noise_drift",
        order=[TASK_LABELS[item] for item in TASK_ORDER],
        color="#2F2F2F",
        alpha=0.48,
        jitter=0.18,
        size=3.0,
        ax=ax,
    )

    summary = df.groupby("task_type", sort=False)["noise_drift"].mean()
    for xpos, task in enumerate(TASK_ORDER):
        value = float(summary.loc[task])
        ax.scatter(
            xpos,
            value,
            marker="D",
            s=34,
            color="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            xpos,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )

    y_max = max(0.28, float(df["noise_drift"].max()) * 1.18)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Task type")
    ax.set_ylabel("Sampling-noise drift")
    ax.set_title("Sampling-Noise Drift Distributions by Task Type")
    ax.grid(axis="y", linestyle="--", color="#D5D5D5", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()

    path = FIGURES / "fig1_noise_baseline.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def create_fig2() -> Path:
    df = pd.read_csv(BASELINE_BY_TASK)
    df["mean"] = df["mean_sampling_noise_drift"].astype(float)
    df["std"] = df["std_sampling_noise_drift"].astype(float)
    df["n"] = df["n_items"].astype(float)
    df["se"] = df["std"] / np.sqrt(df["n"])
    df["ci95"] = 1.96 * df["se"]
    df = df.set_index("task_type").loc[TASK_ORDER].reset_index()

    tukey = pd.read_csv(TUKEY)
    letters = compact_letters(df.sort_values("mean")["task_type"].tolist(), tukey)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y_positions = np.arange(len(df))[::-1]
    for ypos, (_, row) in zip(y_positions, df.iterrows()):
        task = row["task_type"]
        mean_value = float(row["mean"])
        ci95 = float(row["ci95"])
        color = TASK_COLORS[task]
        ax.errorbar(
            mean_value,
            ypos,
            xerr=ci95,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=4,
            markersize=5.5,
        )
        ax.text(
            mean_value + ci95 + 0.006,
            ypos,
            f"{mean_value:.3f} [{letters[task]}]",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([TASK_LABELS[item] for item in TASK_ORDER])
    ax.set_xlabel("Sampling-noise drift")
    ax.set_ylabel("Task type")
    ax.set_title("Tukey HSD Pairwise Comparison Forest Plot")
    ax.grid(axis="x", linestyle="--", color="#D5D5D5", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    x_max = float((df["mean"] + df["ci95"]).max()) + 0.055
    ax.set_xlim(left=0, right=max(0.16, x_max))
    fig.text(
        0.5,
        0.02,
        "Error bars show 95% CIs; groups sharing the same letter are not significantly different (Tukey HSD, alpha=0.05).",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    path = FIGURES / "fig2_tukey_forest.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def create_fig3() -> Path:
    before = load_heatmap(UNCORRECTED_HEATMAP)
    after = load_heatmap(CORRECTED_HEATMAP)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.4),
        gridspec_kw={"wspace": 0.18},
        constrained_layout=False,
    )
    before_cmap = plt.get_cmap("YlOrRd")
    before_vmax = max(0.22, float(before.to_numpy().max()))
    before_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=before_vmax)
    sns.heatmap(
        before,
        ax=axes[0],
        cmap=before_cmap,
        vmin=0,
        vmax=before_vmax,
        annot=False,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Uncorrected drift"},
    )
    annotate_heatmap(axes[0], before, before_cmap, before_norm)
    axes[0].set_title("Before Correction (Uncorrected Drift)", pad=9)
    axes[0].set_xlabel("Perturbation type")
    axes[0].set_ylabel("Task type")

    limit = max(abs(float(after.to_numpy().min())), abs(float(after.to_numpy().max())), 0.05)
    after_cmap = plt.get_cmap("coolwarm")
    after_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    sns.heatmap(
        after,
        ax=axes[1],
        cmap=after_cmap,
        norm=after_norm,
        annot=False,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Noise-corrected drift"},
    )
    annotate_heatmap(axes[1], after, after_cmap, after_norm)
    axes[1].set_title("After Correction (Noise Baseline Removed)", pad=9)
    axes[1].set_xlabel("Perturbation type")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle("Perturbation Drift Heatmap: Before vs After Correction", fontsize=12, y=0.96)
    fig.text(
        0.5,
        0.035,
        "Negative values after correction indicate drift fully explained by sampling noise; values closer to 0 indicate little extra perturbation effect.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.25, top=0.86, wspace=0.25)

    path = FIGURES / "fig3_drift_heatmap.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def simulated_retention_probability(similarity: float, task_type: str) -> float:
    task_adjustment = {
        "factual_qa": 0.04,
        "math_reasoning": -0.08,
        "code_generation": -0.03,
    }[task_type]
    midpoint = 0.72
    slope = 10.5
    probability = 1 / (1 + math.exp(-slope * (similarity - midpoint)))
    probability = 0.18 + 0.78 * probability + task_adjustment
    return min(0.98, max(0.05, probability))


def create_fig4_draft() -> Path:
    source = pd.read_csv(PERTURBATION_BY_ITEM)
    source = source[
        source["task_type"].isin(["factual_qa", "math_reasoning", "code_generation"])
    ].copy()
    source["similarity"] = source["perturbation_similarity"].astype(float)

    rng = np.random.default_rng(20260706)
    source["retention_probability"] = source.apply(
        lambda row: simulated_retention_probability(row["similarity"], row["task_type"]),
        axis=1,
    )
    source["is_correct_simulated"] = (
        rng.random(len(source)) < source["retention_probability"].to_numpy()
    ).astype(int)

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
            retention_rate=("is_correct_simulated", "mean"),
            n=("is_correct_simulated", "size"),
            mean_similarity=("similarity", "mean"),
        )
        .reset_index()
    )
    overall = (
        source.groupby("similarity_bin", observed=False)
        .agg(
            retention_rate=("is_correct_simulated", "mean"),
            n=("is_correct_simulated", "size"),
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

    FIG4_DRAFT_DATA.parent.mkdir(parents=True, exist_ok=True)
    source[
        [
            "item_id",
            "task_type",
            "perturbation_type",
            "similarity",
            "retention_probability",
            "is_correct_simulated",
            "similarity_metric",
        ]
    ].to_csv(FIG4_DRAFT_DATA, index=False)

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
        label="Overall rate (simulated)",
        alpha=0.72,
    )
    for rect, value, count in zip(bars, bar_values, bar_counts):
        if count == 0:
            label = "n=0"
            y = 0.025
        else:
            label = f"{value:.0%}"
            y = value + 0.025
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    for task in ["factual_qa", "math_reasoning", "code_generation"]:
        task_rows = grouped[grouped["task_type"] == task].set_index("similarity_bin")
        y_values = [
            float(task_rows.loc[label, "retention_rate"])
            if int(task_rows.loc[label, "n"]) > 0
            else np.nan
            for label in labels
        ]
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
    ax.set_ylabel("Correctness retention rate")
    ax.set_title("Similarity Bins vs Correctness Retention (Draft Placeholder)")
    ax.grid(axis="y", linestyle="--", color="#D5D5D5", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.text(
        0.5,
        0.02,
        "Note: Llama perturbation samples currently lack true is_correct labels; this draft simulates correctness from the similarity distribution for layout only.",
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


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    require_inputs(
        [
            BASELINE_BY_ITEM,
            BASELINE_BY_TASK,
            TUKEY,
            UNCORRECTED_HEATMAP,
            CORRECTED_HEATMAP,
            PERTURBATION_BY_ITEM,
        ]
    )
    paths = [create_fig1(), create_fig2(), create_fig3(), create_fig4_draft()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
