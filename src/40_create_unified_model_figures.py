"""Create consistently named paper figures for GPT, Llama, and Qwen.

The source data are the final n=50 RQ1/RQ2 outputs already generated in this
project. All images are written to the root figures/ folder with model prefixes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPARISON_FIGURES = ROOT / "figures" / "model_comparison_final"

TASK_ORDER = ["factual_qa", "code_generation", "open_ended_writing", "math_reasoning"]
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
    "formatting_changes": "Formatting",
    "context_injection": "Context injection",
    "surface_noise": "Surface noise",
}

MODELS = [
    {
        "key": "gpt",
        "label": "GPT-4o mini",
        "outputs": ROOT / "outputs",
        "rq2_outputs": ROOT / "rq2_outputs",
        "fig4_real": True,
    },
    {
        "key": "llama",
        "label": "Llama",
        "outputs": ROOT / "data" / "llama" / "outputs",
        "rq2_outputs": ROOT / "data" / "llama" / "rq2_outputs",
        "fig4_real": True,
    },
    {
        "key": "qwen",
        "label": "Qwen",
        "outputs": ROOT / "data" / "qwen" / "outputs",
        "rq2_outputs": ROOT / "data" / "qwen" / "rq2_outputs",
        "fig4_real": True,
    },
]


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "PingFang SC",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 360,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=360, bbox_inches="tight")
    plt.close()
    print(path.relative_to(ROOT))


def annotate_heatmap(ax, data: pd.DataFrame, cmap_name: str, vmin: float, vmax: float, center=None) -> None:
    cmap = plt.get_cmap(cmap_name)
    if center is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        import matplotlib.colors as mcolors

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.52 else "black"
            ax.text(j + 0.5, i + 0.5, f"{value:.3f}", ha="center", va="center", fontsize=7.8, color=color)


def compact_letters(task_df: pd.DataFrame, tukey_df: pd.DataFrame) -> dict[str, str]:
    tasks = task_df.sort_values("mean_sampling_noise_drift")["task_type"].tolist()
    nonsig = {(first, second): first == second for first in tasks for second in tasks}
    for _, row in tukey_df.iterrows():
        first = row["group1"]
        second = row["group2"]
        value = not bool(row["reject_alpha_0_05"])
        nonsig[(first, second)] = value
        nonsig[(second, first)] = value

    letter_sets: list[set[str]] = []
    for task in tasks:
        for group in letter_sets:
            if all(nonsig.get((task, other), False) for other in group):
                group.add(task)
                break
        else:
            letter_sets.append({task})

    for first in tasks:
        for second in tasks:
            if first >= second or not nonsig.get((first, second), False):
                continue
            if any(first in group and second in group for group in letter_sets):
                continue
            candidates = {first, second}
            for task in tasks:
                if task not in candidates and all(nonsig.get((task, other), False) for other in candidates):
                    candidates.add(task)
            letter_sets.append(candidates)

    labels = {task: "" for task in tasks}
    for idx, group in enumerate(letter_sets):
        for task in group:
            labels[task] += "abcdefghijklmnopqrstuvwxyz"[idx]
    return labels


def figure_1(model: dict[str, object]) -> None:
    outputs = Path(model["outputs"])
    df = pd.read_csv(outputs / "sbert_rq1_n50_baseline_by_item.csv")
    df = df.rename(columns={"sampling_noise_drift": "noise_drift"})
    df["task_label"] = df["task_type"].map(TASK_LABELS)
    order_labels = [TASK_LABELS[task] for task in TASK_ORDER]
    palette = [TASK_COLORS[task] for task in TASK_ORDER]

    plt.figure(figsize=(7.2, 4.8))
    ax = sns.violinplot(
        data=df,
        x="task_label",
        y="noise_drift",
        order=order_labels,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        saturation=0.85,
    )
    sns.stripplot(
        data=df,
        x="task_label",
        y="noise_drift",
        order=order_labels,
        palette=palette,
        jitter=0.22,
        size=3.2,
        alpha=0.55,
        linewidth=0.2,
        edgecolor="white",
        ax=ax,
    )
    means = df.groupby("task_type")["noise_drift"].mean()
    for idx, task in enumerate(TASK_ORDER):
        value = means[task]
        ax.scatter(idx, value, marker="D", s=38, color="black", edgecolor="white", linewidth=0.6, zorder=5)
        ax.text(idx, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8.2)

    ax.set_ylim(0, max(0.28, df["noise_drift"].max() + 0.03))
    ax.set_xlabel("Task type")
    ax.set_ylabel("Sampling-noise drift (1 - within-prompt similarity)")
    ax.set_title(f"{model['label']}: Fig. 1. Sampling-noise drift by task type")
    ax.grid(axis="y", color="#D6D6D6", linestyle="--", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(order_labels)))
    ax.set_xticklabels(order_labels, rotation=16, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()
    savefig(MODEL_COMPARISON_FIGURES / f"{model['key']}_fig1_noise_baseline.png")


def figure_2(model: dict[str, object]) -> None:
    outputs = Path(model["outputs"])
    task = pd.read_csv(outputs / "sbert_rq1_n50_baseline_by_task.csv")
    tukey = pd.read_csv(outputs / "rq1_n50_baseline_tukey.csv")
    task = task.set_index("task_type").loc[TASK_ORDER].reset_index()
    task["se"] = task["std_sampling_noise_drift"] / np.sqrt(task["n_items"])
    task["ci95"] = 1.96 * task["se"]
    letters = compact_letters(task, tukey)
    y = np.arange(len(task))
    colors = [TASK_COLORS[t] for t in task["task_type"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.errorbar(
        task["mean_sampling_noise_drift"],
        y,
        xerr=task["ci95"],
        fmt="none",
        ecolor="#4A4A4A",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    ax.scatter(task["mean_sampling_noise_drift"], y, s=55, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for idx, row in task.iterrows():
        mean_value = row["mean_sampling_noise_drift"]
        ax.text(
            mean_value + row["ci95"] + 0.004,
            idx,
            f"{mean_value:.3f} [{letters[row['task_type']]}]",
            va="center",
            ha="left",
            fontsize=8.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABELS[t] for t in task["task_type"]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean sampling-noise drift (95% CI)")
    ax.set_ylabel("Task type")
    ax.set_title(f"{model['label']}: Fig. 2. Baseline differences by task type")
    ax.grid(axis="x", color="#D6D6D6", linestyle="--", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(task["mean_sampling_noise_drift"] + task["ci95"]) + 0.075)
    sns.despine(ax=ax, left=True)
    fig.text(
        0.5,
        0.01,
        "Error bars show 95% CI. Shared letters indicate no significant difference (Tukey HSD, alpha = 0.05).",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    savefig(MODEL_COMPARISON_FIGURES / f"{model['key']}_fig2_tukey_hsd.png")


def heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    mat = df.set_index("perturbation_type").T
    mat = mat.loc[TASK_ORDER, PERTURBATION_ORDER]
    mat.index = [TASK_LABELS[t] for t in mat.index]
    mat.columns = [PERTURBATION_LABELS[p] for p in mat.columns]
    return mat.astype(float)


def figure_3(model: dict[str, object]) -> None:
    outputs = Path(model["outputs"])
    before_mat = heatmap_matrix(pd.read_csv(outputs / "sbert_rq1_n50_uncorrected_heatmap_drift.csv"))
    after_mat = heatmap_matrix(pd.read_csv(outputs / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    before_max = float(before_mat.max().max())
    sns.heatmap(
        before_mat,
        ax=axes[0],
        cmap="YlOrRd",
        vmin=0,
        vmax=before_max,
        cbar_kws={"label": "Mean uncorrected drift"},
        linewidths=0.4,
        linecolor="white",
        annot=False,
    )
    annotate_heatmap(axes[0], before_mat, "YlOrRd", 0, before_max)
    axes[0].set_title("Before correction: raw semantic drift")
    axes[0].set_xlabel("Perturbation type")
    axes[0].set_ylabel("Task type")

    lim = float(np.nanmax(np.abs(after_mat.to_numpy())))
    sns.heatmap(
        after_mat,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        vmin=-lim,
        vmax=lim,
        cbar_kws={"label": "Mean noise-corrected drift"},
        linewidths=0.4,
        linecolor="white",
        annot=False,
    )
    annotate_heatmap(axes[1], after_mat, "RdBu_r", -lim, lim, center=0)
    axes[1].set_title("After correction: drift beyond noise baseline")
    axes[1].set_xlabel("Perturbation type")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.suptitle(f"{model['label']}: Fig. 3. Perturbation drift before and after baseline correction", fontsize=12.5, y=1.04)
    fig.text(
        0.5,
        -0.03,
        "Negative or near-zero corrected values indicate effects explained by sampling noise; larger positive values indicate additional perturbation effect.",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#444444",
    )
    savefig(MODEL_COMPARISON_FIGURES / f"{model['key']}_fig3_drift_heatmap.png")


def figure_4(model: dict[str, object]) -> None:
    rq2_outputs = Path(model["rq2_outputs"])
    if bool(model["fig4_real"]):
        source = rq2_outputs / "fig4_similarity_correctness_data.csv"
        if not source.exists():
            print(f"Skipping {model['key']} fig4: missing {source}")
            return
        df = pd.read_csv(source)
        correctness_col = "performance_retention"
        filename = f"{model['key']}_fig4_similarity_performance_retention.png"
        note = "Retention is computed as perturbed performance divided by original performance, capped at 100%; cases with zero original performance are excluded."
        title_suffix = "RQ2 performance data"
    else:
        source = rq2_outputs / "rq2_formal_available_drift_performance_by_item.csv"
        if not source.exists():
            print(f"Skipping {model['key']} fig4: missing {source}")
            return
        df = pd.read_csv(source)
        df["similarity"] = df["perturbation_similarity"].astype(float)
        correctness_col = "perturbed_performance"
        filename = f"{model['key']}_fig4_similarity_performance_retention_draft.png"
        note = "Draft: uses available item-level RQ2 performance data, not generation-level formal RQ2 labels."
        title_suffix = "draft item-level RQ2 data"

    rq2_task_order = ["factual_qa", "math_reasoning", "code_generation"]
    df = df[df["task_type"].isin(rq2_task_order)].copy()
    df["similarity"] = df["similarity"].astype(float)
    df["original_performance"] = df["original_performance"].astype(float).clip(lower=0)
    if "correctness_retention" in df.columns:
        df["performance_retention"] = pd.to_numeric(df["correctness_retention"], errors="coerce").clip(0, 1)
    else:
        df["perturbed_performance"] = df[correctness_col].astype(float).clip(0, 1)
        df = df[df["original_performance"] > 0].copy()
        df["performance_retention"] = (df["perturbed_performance"] / df["original_performance"]).clip(upper=1)
    df = df.dropna(subset=["performance_retention"]).copy()
    bins = [-np.inf, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001]
    labels = ["<0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
    df["similarity_bin"] = pd.cut(df["similarity"], bins=bins, labels=labels, include_lowest=True, right=False)
    ascending_summary = (
        df.groupby("similarity_bin", observed=False)
        .agg(performance_retention=("performance_retention", "mean"), n=("performance_retention", "size"))
        .reset_index()
    )
    ascending_summary = ascending_summary[ascending_summary["n"] > 0].copy()

    high_to_low = list(reversed(labels))
    summary = ascending_summary.set_index("similarity_bin").reindex(high_to_low).dropna(subset=["performance_retention"])
    task_summary = (
        df.groupby(["similarity_bin", "task_type"], observed=False)
        .agg(performance_retention=("performance_retention", "mean"), n=("performance_retention", "size"))
        .reset_index()
    )

    min_task_bin_n = 5
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    bars = ax.bar(
        x,
        summary["performance_retention"] * 100,
        color="#C9C9C9",
        edgecolor="#8C8C8C",
        linewidth=0.8,
        alpha=0.88,
        label="Overall correctness retention",
        zorder=1,
    )

    for task in rq2_task_order:
        task_series = (
            task_summary[task_summary["task_type"] == task]
            .set_index("similarity_bin")
            .reindex(summary.index)
        )
        y = task_series["performance_retention"].astype(float) * 100
        n_values = task_series["n"].fillna(0).astype(float)
        reliable = n_values >= min_task_bin_n
        y_line = y.where(reliable)
        ax.plot(
            x,
            y_line,
            color=TASK_COLORS[task],
            marker=None,
            linewidth=1.8,
            label=TASK_LABELS[task],
            zorder=3,
        )
        ax.scatter(
            x[reliable.to_numpy()],
            y[reliable].to_numpy(),
            color=TASK_COLORS[task],
            s=34,
            zorder=4,
        )
        ax.scatter(
            x[(~reliable).to_numpy()],
            y[~reliable].to_numpy(),
            facecolors="white",
            edgecolors=TASK_COLORS[task],
            linewidths=1.5,
            s=42,
            zorder=5,
        )

    for bar, (_, row) in zip(bars, summary.iterrows()):
        y_value = row["performance_retention"] * 100
        y_label = min(y_value + 2.0, 101.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_label,
            f"{row['performance_retention'] * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in summary.index], rotation=25, ha="right")
    ax.set_ylim(0, 112)
    ax.set_xlabel("Output similarity bin (high to low)")
    ax.set_ylabel("Correctness retention (%)")
    ax.set_title(
        f"{model['label']}: Fig. 4. Similarity bins vs. correctness retention ({title_suffix})",
        pad=32,
    )
    ax.grid(axis="y", color="#D6D6D6", linestyle="--", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    legend_handles = [
        Line2D([0], [0], color=TASK_COLORS["factual_qa"], linewidth=1.8, label=TASK_LABELS["factual_qa"]),
        Line2D([0], [0], color=TASK_COLORS["math_reasoning"], linewidth=1.8, label=TASK_LABELS["math_reasoning"]),
        Line2D([0], [0], color=TASK_COLORS["code_generation"], linewidth=1.8, label=TASK_LABELS["code_generation"]),
        Patch(facecolor="#C9C9C9", edgecolor="#8C8C8C", label="Overall correctness retention"),
        Line2D([0], [0], marker="o", color="#333333", linestyle="None", markersize=5.2, label=f"Solid marker: n >= {min_task_bin_n}"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#333333",
            linestyle="None",
            markersize=5.8,
            label=f"Hollow marker: n < {min_task_bin_n}",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        columnspacing=1.2,
        handlelength=1.9,
    )
    ax.text(
        0,
        -0.31,
        note + "\n" + f"Hollow markers indicate task-specific bins with n < {min_task_bin_n}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color="#555555",
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    savefig(MODEL_COMPARISON_FIGURES / filename)


def remove_old_top_level_model_figures() -> None:
    old_patterns = [
        "fig1_noise_baseline.png",
        "fig2_tukey_forest.png",
        "fig3_drift_heatmap.png",
        "fig4_similarity_correctness_draft.png",
        "llama_fig1_noise_baseline.png",
        "llama_fig2_tukey_forest.png",
        "llama_fig3_drift_heatmap.png",
        "llama_fig4_similarity_correctness.png",
        "llama_fig4_similarity_correctness_draft.png",
        "qwen_fig1_noise_baseline.png",
        "qwen_fig2_tukey_forest.png",
        "qwen_fig3_drift_heatmap.png",
        "qwen_fig4_similarity_correctness.png",
        "qwen_fig4_similarity_correctness_draft.png",
    ]
    for name in old_patterns:
        path = MODEL_COMPARISON_FIGURES / name
        if path.exists():
            path.unlink()


def main() -> None:
    MODEL_COMPARISON_FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    remove_old_top_level_model_figures()
    for model in MODELS:
        figure_1(model)
        figure_2(model)
        figure_3(model)
        figure_4(model)


if __name__ == "__main__":
    main()
