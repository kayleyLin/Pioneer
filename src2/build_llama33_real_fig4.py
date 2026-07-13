"""Build real Fig. 4 correctness-retention data for Llama 3.3 70B n=50 outputs.

This follows the same calculation used for figures_new:
- factual QA: reference-answer containment, otherwise token-F1 performance
- math reasoning: final-answer equivalence, scored 0/1
- code generation: HumanEvalPack functional tests, scored 0/1
- retention: min(perturbed_performance / original_performance, 1.0)

Rows with original_performance == 0 have blank retention and are excluded from
the binned means.
"""

from __future__ import annotations

import csv
import importlib.util
import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "llama" / "llama33_70b_instruct_turbo_150case"
OUTPUTS = BASE / "outputs"
RQ2_OUTPUTS = BASE / "rq2_outputs"
FIGURES = BASE / "figures"
PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50.csv"
ORIGINAL = OUTPUTS / "rq1_llama33_70b_original_generations_n50_four_task.csv"
PERTURBED = OUTPUTS / "rq1_llama33_70b_perturbed_generations_n50_four_task.csv"
EFFECTS = OUTPUTS / "sbert_rq1_n50_perturbation_effects_by_item.csv"

RQ2_TASKS = {"factual_qa", "math_reasoning", "code_generation"}
TASK_ORDER = ["factual_qa", "math_reasoning", "code_generation"]
TASK_LABELS = {
    "factual_qa": "Factual QA",
    "math_reasoning": "Math reasoning",
    "code_generation": "Code generation",
}
TASK_COLORS = {
    "factual_qa": "#4C72B0",
    "math_reasoning": "#DD8452",
    "code_generation": "#55A868",
}


def load_evaluator():
    path = ROOT / "src" / "23_evaluate_rq2_correctness.py"
    spec = importlib.util.spec_from_file_location("rq2_correctness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_humanevalpack_tests_from_cache(source_indices: set[str]) -> dict[str, dict[str, str]]:
    arrow_path = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "bigcode___humanevalpack"
        / "python"
        / "0.0.0"
        / "9a41762f73a8cb23bb5811b73d5aab164efcf378"
        / "humanevalpack-test.arrow"
    )
    if not arrow_path.exists():
        raise SystemExit(f"Missing cached HumanEvalPack arrow file: {arrow_path}")

    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.open_stream(source).read_all()

    rows = table.to_pylist()
    tests: dict[str, dict[str, str]] = {}
    for source_index in source_indices:
        try:
            row = rows[int(source_index)]
        except Exception:
            continue
        tests[source_index] = {
            "task_id": str(row.get("task_id", "")),
            "entry_point": str(row.get("entry_point", "")),
            "declaration": str(row.get("declaration", "")),
            "imports": str(row.get("import", "")),
            "test_setup": str(row.get("test_setup", "")),
            "test": str(row.get("test", "")),
        }
    return tests


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    if fieldnames is None:
        raise ValueError(f"No fieldnames for empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_inputs() -> None:
    missing = [str(path) for path in [PROMPTS, ORIGINAL, PERTURBED, EFFECTS] if not path.exists()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))


def evaluate_generations(
    evaluator,
    generation_rows: list[dict[str, str]],
    prompt_by_item: dict[str, dict[str, str]],
    prompt_variant: str,
) -> list[dict[str, str]]:
    code_source_indices = {
        row.get("source_index", "")
        for row in generation_rows
        if row.get("task_type") == "code_generation"
    }
    code_tests = load_humanevalpack_tests_from_cache(code_source_indices)

    def evaluate_one(index: int, row: dict[str, str]) -> tuple[int, dict[str, str]]:
        prompt_meta = prompt_by_item.get(row["item_id"], {})
        result = evaluator.evaluate_row(row, prompt_meta, code_tests)
        return index, (
            {
                "item_id": row.get("item_id", ""),
                "task_type": row.get("task_type", ""),
                "dataset_name": row.get("dataset_name", ""),
                "source_index": row.get("source_index", ""),
                "source_id": prompt_meta.get("source_id", ""),
                "sample_id": row.get("sample_id", ""),
                "model_name": row.get("model_name", ""),
                "prompt_variant": prompt_variant,
                "perturbation_type": row.get("perturbation_type", ""),
                "reference_answer": prompt_meta.get("reference_answer", ""),
                "output_text": row.get("output_text", ""),
                "extracted_answer": result.extracted_answer,
                "performance_score": ""
                if result.performance_score is None
                else f"{result.performance_score:.6f}",
                "factual_containment_match": ""
                if result.factual_containment_match is None
                else ("true" if result.factual_containment_match else "false"),
                "factual_token_f1": ""
                if result.factual_token_f1 is None
                else f"{result.factual_token_f1:.6f}",
                "is_correct": evaluator.bool_to_csv(result.is_correct),
                "needs_manual_review": "true" if result.needs_manual_review else "false",
                "correctness_method": result.correctness_method,
                "notes": result.notes,
            }
        )

    output_rows: list[dict[str, str] | None] = [None] * len(generation_rows)
    completed = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(evaluate_one, index, row): index
            for index, row in enumerate(generation_rows)
        }
        for future in as_completed(futures):
            index, output_row = future.result()
            output_rows[index] = output_row
            completed += 1
            if completed % 250 == 0:
                print(
                    f"Evaluated {completed}/{len(generation_rows)} {prompt_variant} rows",
                    flush=True,
                )
    return [row for row in output_rows if row is not None]


def score(row: dict[str, str]) -> float:
    value = row.get("performance_score", "")
    if value == "":
        raise ValueError(f"Missing performance_score for {row.get('item_id')}")
    return float(value)


def mean_score(rows: list[dict[str, str]]) -> float:
    return mean(score(row) for row in rows)


def safe_pdr(original_score: float, perturbed_score: float) -> str:
    if original_score == 0:
        return ""
    return f"{(original_score - perturbed_score) / original_score:.6f}"


def build_performance_change(
    original_rows: list[dict[str, str]],
    perturbed_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    original_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in original_rows:
        original_by_item[row["item_id"]].append(row)

    perturbed_by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(row)

    item_rows = []
    for (item_id, perturbation_type), rows in sorted(perturbed_by_group.items()):
        if item_id not in original_by_item:
            continue
        original_score = mean_score(original_by_item[item_id])
        perturbed_score = mean_score(rows)
        absolute_change = original_score - perturbed_score
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": rows[0]["task_type"],
                "perturbation_type": perturbation_type,
                "n_original_outputs": str(len(original_by_item[item_id])),
                "n_perturbed_outputs": str(len(rows)),
                "original_performance": f"{original_score:.6f}",
                "perturbed_performance": f"{perturbed_score:.6f}",
                "absolute_performance_change": f"{absolute_change:.6f}",
                "pdr": safe_pdr(original_score, perturbed_score),
                "performance_dropped": "true" if perturbed_score < original_score else "false",
            }
        )
    return item_rows


def build_drift_performance(performance_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    effects = pd.read_csv(EFFECTS)
    effect_by_group = {
        (row.item_id, row.perturbation_type): row
        for row in effects.itertuples(index=False)
        if row.task_type in RQ2_TASKS
    }

    out_rows = []
    for row in performance_rows:
        key = (row["item_id"], row["perturbation_type"])
        if key not in effect_by_group:
            continue
        effect = effect_by_group[key]
        out_rows.append(
            {
                **row,
                "baseline_similarity": f"{float(effect.baseline_similarity):.6f}",
                "perturbation_similarity": f"{float(effect.perturbation_similarity):.6f}",
                "uncorrected_drift": f"{float(effect.uncorrected_drift):.6f}",
                "noise_corrected_drift": f"{float(effect.noise_corrected_drift):.6f}",
                "similarity_metric": str(effect.similarity_metric),
            }
        )
    return out_rows


def build_fig4_data(drift_performance_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    fig_rows = []
    for row in drift_performance_rows:
        original_performance = float(row["original_performance"])
        perturbed_performance = float(row["perturbed_performance"])
        if original_performance > 0:
            correctness_retention = min(perturbed_performance / original_performance, 1.0)
            correctness_retention_text = f"{correctness_retention:.6f}"
        else:
            correctness_retention_text = ""
        fig_rows.append(
            {
                "item_id": row["item_id"],
                "task_type": row["task_type"],
                "perturbation_type": row["perturbation_type"],
                "similarity": row["perturbation_similarity"],
                "perturbed_performance": row["perturbed_performance"],
                "original_performance": row["original_performance"],
                "correctness_retention": correctness_retention_text,
                "absolute_performance_change": row["absolute_performance_change"],
                "pdr": row["pdr"],
                "similarity_metric": row["similarity_metric"],
            }
        )

    df = pd.DataFrame(fig_rows)
    df["similarity"] = df["similarity"].astype(float)
    df["correctness_retention"] = pd.to_numeric(
        df["correctness_retention"], errors="coerce"
    ).clip(0, 1)
    bins = [-float("inf"), 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001]
    labels = [
        "<0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]
    df["similarity_bin"] = pd.cut(
        df["similarity"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    valid = df.dropna(subset=["correctness_retention"]).copy()
    overall = (
        valid.groupby("similarity_bin", observed=False)
        .agg(
            mean_correctness_retention=("correctness_retention", "mean"),
            n=("correctness_retention", "size"),
        )
        .reset_index()
    )
    overall = overall[overall["n"] > 0].copy()
    by_task = (
        valid.groupby(["similarity_bin", "task_type"], observed=False)
        .agg(
            mean_correctness_retention=("correctness_retention", "mean"),
            n=("correctness_retention", "size"),
        )
        .reset_index()
    )
    by_task = by_task[by_task["n"] > 0].copy()

    overall_rows = [
        {
            "similarity_bin": str(row.similarity_bin),
            "mean_correctness_retention": f"{float(row.mean_correctness_retention):.6f}",
            "n": str(int(row.n)),
        }
        for row in overall.itertuples(index=False)
    ]
    by_task_rows = [
        {
            "similarity_bin": str(row.similarity_bin),
            "task_type": str(row.task_type),
            "mean_correctness_retention": f"{float(row.mean_correctness_retention):.6f}",
            "n": str(int(row.n)),
        }
        for row in by_task.itertuples(index=False)
    ]
    return fig_rows, overall_rows, by_task_rows


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


def plot_fig4() -> Path:
    df = pd.read_csv(RQ2_OUTPUTS / "fig4_similarity_correctness_data.csv")
    df = df[df["task_type"].isin(TASK_ORDER)].copy()
    df["similarity"] = df["similarity"].astype(float)
    df["original_performance"] = df["original_performance"].astype(float).clip(lower=0)
    df["performance_retention"] = pd.to_numeric(
        df["correctness_retention"], errors="coerce"
    ).clip(0, 1)
    df = df.dropna(subset=["performance_retention"]).copy()

    bins = [-np.inf, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001]
    labels = [
        "<0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]
    df["similarity_bin"] = pd.cut(
        df["similarity"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    ascending_summary = (
        df.groupby("similarity_bin", observed=False)
        .agg(
            performance_retention=("performance_retention", "mean"),
            n=("performance_retention", "size"),
        )
        .reset_index()
    )
    ascending_summary = ascending_summary[ascending_summary["n"] > 0].copy()
    high_to_low = list(reversed(labels))
    summary = (
        ascending_summary.set_index("similarity_bin")
        .reindex(high_to_low)
        .dropna(subset=["performance_retention"])
    )
    task_summary = (
        df.groupby(["similarity_bin", "task_type"], observed=False)
        .agg(
            performance_retention=("performance_retention", "mean"),
            n=("performance_retention", "size"),
        )
        .reset_index()
    )

    min_task_bin_n = 5
    min_overall_bin_n = 20
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    reliable_overall = summary["n"].astype(float) >= min_overall_bin_n
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
    for bar, is_reliable in zip(bars, reliable_overall):
        if not bool(is_reliable):
            bar.set_hatch("///")
            bar.set_alpha(0.48)

    for task in TASK_ORDER:
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
            f"{row['performance_retention'] * 100:.0f}%\nn={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in summary.index], rotation=25, ha="right")
    ax.set_ylim(0, 112)
    ax.set_xlabel("Output similarity bin (high to low)")
    ax.set_ylabel("Correctness retention (%)")
    ax.set_title(
        "Llama 3.3 70B: Fig. 4. Similarity bins vs. correctness retention",
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
        Patch(
            facecolor="#C9C9C9",
            edgecolor="#8C8C8C",
            hatch="///",
            alpha=0.48,
            label=f"Overall bin n < {min_overall_bin_n}",
        ),
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
        -0.33,
        "Retention is perturbed performance divided by original performance, capped at 100%; rows with zero original performance are excluded.\n"
        f"Hatched bars indicate low overall bin counts (n < {min_overall_bin_n}); hollow markers indicate task-specific bins with n < {min_task_bin_n}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color="#555555",
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "fig4_similarity_performance_retention.png"
    fig.savefig(path, dpi=360, bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_similarity_correctness.png", dpi=360, bbox_inches="tight")
    plt.close(fig)
    return path


def validate() -> None:
    fig4 = pd.read_csv(RQ2_OUTPUTS / "fig4_similarity_correctness_data.csv")
    valid = pd.to_numeric(fig4["correctness_retention"], errors="coerce").notna().sum()
    if len(fig4) != 750:
        raise SystemExit(f"Expected 750 fig4 rows, got {len(fig4)}")
    if valid == 0:
        raise SystemExit("No valid correctness_retention rows")
    task_counts = fig4.groupby("task_type")["item_id"].nunique().to_dict()
    expected = {task: 50 for task in TASK_ORDER}
    if task_counts != expected:
        raise SystemExit(f"Unexpected task item counts: {task_counts}")


def main() -> None:
    global RQ2_OUTPUTS, FIGURES

    parser = argparse.ArgumentParser()
    parser.add_argument("--rq2-output-dir", type=Path, default=RQ2_OUTPUTS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    RQ2_OUTPUTS = args.rq2_output_dir
    FIGURES = args.figure_dir

    require_inputs()
    RQ2_OUTPUTS.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator()
    prompt_rows = read_csv(PROMPTS)
    prompt_by_item = {row["item_id"]: row for row in prompt_rows}

    original_rows = [row for row in read_csv(ORIGINAL) if row.get("task_type") in RQ2_TASKS]
    perturbed_rows = [row for row in read_csv(PERTURBED) if row.get("task_type") in RQ2_TASKS]

    original_correctness = evaluate_generations(evaluator, original_rows, prompt_by_item, "original")
    perturbed_correctness = evaluate_generations(evaluator, perturbed_rows, prompt_by_item, "perturbed")
    correctness_fieldnames = [
        "item_id",
        "task_type",
        "dataset_name",
        "source_index",
        "source_id",
        "sample_id",
        "model_name",
        "prompt_variant",
        "perturbation_type",
        "reference_answer",
        "output_text",
        "extracted_answer",
        "performance_score",
        "factual_containment_match",
        "factual_token_f1",
        "is_correct",
        "needs_manual_review",
        "correctness_method",
        "notes",
    ]
    write_csv(RQ2_OUTPUTS / "rq2_original_correctness_by_generation.csv", original_correctness, correctness_fieldnames)
    write_csv(RQ2_OUTPUTS / "rq2_perturbed_correctness_by_generation.csv", perturbed_correctness, correctness_fieldnames)
    write_csv(RQ2_OUTPUTS / "rq2_original_correctness_summary_by_task.csv", evaluator.summarize(original_correctness))
    write_csv(RQ2_OUTPUTS / "rq2_perturbed_correctness_summary_by_task.csv", evaluator.summarize(perturbed_correctness))

    performance_change = build_performance_change(original_correctness, perturbed_correctness)
    write_csv(RQ2_OUTPUTS / "rq2_performance_change_by_item.csv", performance_change)
    drift_performance = build_drift_performance(performance_change)
    write_csv(RQ2_OUTPUTS / "rq2_drift_performance_by_item.csv", drift_performance)
    fig4_rows, fig4_summary, fig4_by_task = build_fig4_data(drift_performance)
    write_csv(RQ2_OUTPUTS / "fig4_similarity_correctness_data.csv", fig4_rows)
    write_csv(RQ2_OUTPUTS / "fig4_similarity_correctness_binned.csv", fig4_summary)
    write_csv(RQ2_OUTPUTS / "fig4_similarity_correctness_binned_by_task.csv", fig4_by_task)

    configure_style()
    fig_path = plot_fig4()
    validate()
    print(f"Wrote RQ2 outputs to {RQ2_OUTPUTS}")
    print(f"Original correctness rows: {len(original_correctness)}")
    print(f"Perturbed correctness rows: {len(perturbed_correctness)}")
    print(f"Figure 4 rows: {len(fig4_rows)}")
    print(fig_path)


if __name__ == "__main__":
    main()
