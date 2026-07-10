"""Run RQ1 n=50 Sentence-BERT analysis for external model result folders.

Usage:
    python src/37_analyze_rq1_external_model_sbert.py --model-dir data/llama --prefix rq1_llama
    python src/37_analyze_rq1_external_model_sbert.py --model-dir data/qwen --prefix rq1_qwen

This script reads already-generated original and perturbed outputs. It does not
call any LLM API.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from statsmodels.stats.multicomp import pairwise_tukeyhsd


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TASKS = ["factual_qa", "math_reasoning", "code_generation", "open_ended_writing"]
TASK_ORDER = ["code_generation", "factual_qa", "math_reasoning", "open_ended_writing"]
PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


class SimilarityModel:
    def __init__(self) -> None:
        print(f"Loading Sentence-BERT model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.cache = {}

    def encode(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self.cache[text]

    def similarity(self, text_a: str, text_b: str) -> float:
        return float(cos_sim(self.encode(text_a), self.encode(text_b))[0][0])

    def within_similarity(self, outputs: list[str]) -> float:
        return mean(
            [
                self.similarity(output_a, output_b)
                for output_a, output_b in combinations(outputs, 2)
            ]
        )

    def cross_similarity(self, original_outputs: list[str], perturbed_outputs: list[str]) -> float:
        return mean(
            [
                self.similarity(original_output, perturbed_output)
                for original_output, perturbed_output in product(
                    original_outputs, perturbed_outputs
                )
            ]
        )


def resolve_model_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def generation_files(model_dir: Path, prefix: str, variant: str) -> list[Path]:
    return [
        model_dir / "outputs" / f"{prefix}_{variant}_generations_n50_{task}.csv"
        for task in TASKS
    ]


def load_all(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")
        file_rows = read_csv(path)
        print(f"Loaded {len(file_rows)} rows from {path}")
        rows.extend(file_rows)
    return rows


def validate_original_rows(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, int] = defaultdict(int)
    empty = 0
    for row in rows:
        grouped[row["item_id"]] += 1
        if not row.get("output_text", "").strip():
            empty += 1
    bad = {item_id: count for item_id, count in grouped.items() if count != 5}
    if empty:
        raise SystemExit(f"Found {empty} empty original output rows.")
    if bad:
        raise SystemExit(f"Original rows with non-5 samples: {list(bad.items())[:10]}")
    print(f"Validated {len(grouped)} original prompts with 5 outputs each.")


def validate_perturbed_rows(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    empty = 0
    for row in rows:
        grouped[(row["item_id"], row["perturbation_type"])] += 1
        if not row.get("output_text", "").strip():
            empty += 1
    bad = {key: count for key, count in grouped.items() if count != 5}
    if empty:
        raise SystemExit(f"Found {empty} empty perturbed output rows.")
    if bad:
        raise SystemExit(f"Perturbed rows with non-5 samples: {list(bad.items())[:10]}")
    print(f"Validated {len(grouped)} item-perturbation pairs with 5 outputs each.")


def build_heatmap_rows(summary_rows: list[dict], value_key: str) -> list[dict]:
    task_types = sorted({row["task_type"] for row in summary_rows})
    lookup = {
        (row["task_type"], row["perturbation_type"]): row[value_key]
        for row in summary_rows
    }
    rows = []
    for perturbation_type in PERTURBATION_ORDER:
        row = {"perturbation_type": perturbation_type}
        for task_type in task_types:
            row[task_type] = lookup.get((task_type, perturbation_type), "")
        rows.append(row)
    return rows


def analyze(model_dir: Path, prefix: str) -> None:
    outputs_dir = model_dir / "outputs"
    figures_dir = model_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    original_rows = load_all(generation_files(model_dir, prefix, "original"))
    perturbed_rows = load_all(generation_files(model_dir, prefix, "perturbed"))
    validate_original_rows(original_rows)
    validate_perturbed_rows(perturbed_rows)

    sim_model = SimilarityModel()

    original_by_item: dict[str, list[str]] = defaultdict(list)
    task_by_item: dict[str, str] = {}
    dataset_by_item: dict[str, str] = {}
    for row in original_rows:
        original_by_item[row["item_id"]].append(row["output_text"])
        task_by_item[row["item_id"]] = row["task_type"]
        dataset_by_item[row["item_id"]] = row["dataset_name"]

    baseline_by_item = {}
    baseline_item_rows = []
    task_drifts: dict[str, list[float]] = defaultdict(list)
    task_similarities: dict[str, list[float]] = defaultdict(list)

    for item_id, outputs in sorted(original_by_item.items()):
        baseline_similarity = sim_model.within_similarity(outputs)
        drift = 1 - baseline_similarity
        baseline_by_item[item_id] = baseline_similarity
        task_type = task_by_item[item_id]
        baseline_item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "dataset_name": dataset_by_item[item_id],
                "n_outputs": len(outputs),
                "n_pairs": len(outputs) * (len(outputs) - 1) // 2,
                "mean_within_prompt_similarity": round(baseline_similarity, 6),
                "sampling_noise_drift": round(drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )
        task_drifts[task_type].append(drift)
        task_similarities[task_type].append(baseline_similarity)

    baseline_task_rows = []
    for task_type in sorted(task_drifts):
        baseline_task_rows.append(
            {
                "task_type": task_type,
                "n_items": len(task_drifts[task_type]),
                "mean_within_prompt_similarity": round(mean(task_similarities[task_type]), 6),
                "mean_sampling_noise_drift": round(mean(task_drifts[task_type]), 6),
                "std_sampling_noise_drift": round(sample_std(task_drifts[task_type]), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    write_csv(outputs_dir / "sbert_rq1_n50_baseline_by_item.csv", baseline_item_rows)
    write_csv(outputs_dir / "sbert_rq1_n50_baseline_by_task.csv", baseline_task_rows)

    baseline_df = pd.DataFrame(baseline_item_rows)
    groups = [
        group["sampling_noise_drift"].astype(float).to_numpy()
        for _, group in baseline_df.groupby("task_type")
    ]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    levene = stats.levene(*groups, center="median")
    write_csv(
        outputs_dir / "rq1_n50_baseline_significance_tests.csv",
        [
            {
                "test": "one_way_anova",
                "statistic": round(float(anova.statistic), 6),
                "p_value": round(float(anova.pvalue), 12),
                "interpretation": "tests mean baseline drift differences across task types",
            },
            {
                "test": "kruskal_wallis",
                "statistic": round(float(kruskal.statistic), 6),
                "p_value": round(float(kruskal.pvalue), 12),
                "interpretation": "nonparametric check for baseline drift differences across task types",
            },
            {
                "test": "levene_median",
                "statistic": round(float(levene.statistic), 6),
                "p_value": round(float(levene.pvalue), 12),
                "interpretation": "checks equality of variance across task types",
            },
        ],
    )
    tukey = pairwise_tukeyhsd(
        endog=baseline_df["sampling_noise_drift"].astype(float),
        groups=baseline_df["task_type"],
        alpha=0.05,
    )
    tukey_rows = []
    for row in tukey.summary().data[1:]:
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        tukey_rows.append(
            {
                "group1": group1,
                "group2": group2,
                "mean_difference": round(float(meandiff), 6),
                "p_adj": round(float(p_adj), 12),
                "ci_lower": round(float(lower), 6),
                "ci_upper": round(float(upper), 6),
                "reject_alpha_0_05": bool(reject),
            }
        )
    write_csv(outputs_dir / "rq1_n50_baseline_tukey.csv", tukey_rows)

    perturbed_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(
            row["output_text"]
        )

    effect_rows = []
    for (item_id, perturbation_type), perturbed_outputs in sorted(perturbed_by_group.items()):
        original_outputs = original_by_item[item_id]
        baseline_similarity = baseline_by_item[item_id]
        perturbation_similarity = sim_model.cross_similarity(original_outputs, perturbed_outputs)
        effect_rows.append(
            {
                "item_id": item_id,
                "task_type": task_by_item[item_id],
                "perturbation_type": perturbation_type,
                "n_original_outputs": len(original_outputs),
                "n_perturbed_outputs": len(perturbed_outputs),
                "baseline_similarity": round(baseline_similarity, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "uncorrected_drift": round(1 - perturbation_similarity, 6),
                "noise_corrected_drift": round(
                    baseline_similarity - perturbation_similarity, 6
                ),
                "similarity_metric": MODEL_NAME,
            }
        )

    write_csv(outputs_dir / "sbert_rq1_n50_perturbation_effects_by_item.csv", effect_rows)

    summary_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in effect_rows:
        summary_groups[(row["task_type"], row["perturbation_type"])].append(row)

    summary_rows = []
    uncorrected_rows = []
    for (task_type, perturbation_type), rows in sorted(summary_groups.items()):
        corrected = [float(row["noise_corrected_drift"]) for row in rows]
        uncorrected = [float(row["uncorrected_drift"]) for row in rows]
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(rows),
                "mean_noise_corrected_drift": round(mean(corrected), 6),
                "std_noise_corrected_drift": round(sample_std(corrected), 6),
                "similarity_metric": MODEL_NAME,
            }
        )
        uncorrected_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(rows),
                "mean_uncorrected_drift": round(mean(uncorrected), 6),
                "std_uncorrected_drift": round(sample_std(uncorrected), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    task_types = sorted({row["task_type"] for row in summary_rows})
    write_csv(outputs_dir / "sbert_rq1_n50_perturbation_summary.csv", summary_rows)
    write_csv(
        outputs_dir / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv",
        build_heatmap_rows(summary_rows, "mean_noise_corrected_drift"),
        ["perturbation_type", *task_types],
    )
    write_csv(
        outputs_dir / "sbert_rq1_n50_uncorrected_perturbation_summary.csv",
        uncorrected_rows,
    )
    write_csv(
        outputs_dir / "sbert_rq1_n50_uncorrected_heatmap_drift.csv",
        build_heatmap_rows(uncorrected_rows, "mean_uncorrected_drift"),
        ["perturbation_type", *task_types],
    )

    corrected = pd.read_csv(outputs_dir / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv")
    uncorrected = pd.read_csv(outputs_dir / "sbert_rq1_n50_uncorrected_heatmap_drift.csv")
    create_heatmaps(corrected, uncorrected, figures_dir)

    print(f"Finished analysis for {model_dir}")


def create_heatmaps(corrected: pd.DataFrame, uncorrected: pd.DataFrame, figures_dir: Path) -> None:
    def matrix(df: pd.DataFrame) -> pd.DataFrame:
        mat = df.set_index("perturbation_type").reindex(PERTURBATION_ORDER)
        mat = mat[[col for col in TASK_ORDER if col in mat.columns]]
        return mat.astype(float)

    corrected_mat = matrix(corrected)
    uncorrected_mat = matrix(uncorrected)

    for data, filename, title, label, cmap, center in [
        (
            uncorrected_mat,
            "rq1_n50_uncorrected_drift_heatmap.png",
            "RQ1 n=50: Uncorrected Semantic Drift",
            "Mean uncorrected drift",
            "YlOrRd",
            None,
        ),
        (
            corrected_mat,
            "rq1_n50_noise_corrected_drift_heatmap.png",
            "RQ1 n=50: Noise-Corrected Semantic Drift",
            "Mean noise-corrected drift",
            "vlag",
            0.0,
        ),
    ]:
        plt.figure(figsize=(8.8, 5.2))
        ax = sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            center=center,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": label},
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Task type")
        ax.set_ylabel("Perturbation type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        path = figures_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    analyze(resolve_model_dir(args.model_dir), args.prefix)


if __name__ == "__main__":
    main()
