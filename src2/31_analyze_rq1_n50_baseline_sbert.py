"""Analyze the expanded RQ1 n=50 sample-noise baseline with Sentence-BERT.

Inputs:
    outputs/rq1_formal_original_generations_n50_factual_qa.csv
    outputs/rq1_formal_original_generations_n50_math_reasoning.csv
    outputs/rq1_formal_original_generations_n50_code_generation.csv
    outputs/rq1_formal_original_generations_n50_open_ended_writing.csv

Outputs:
    outputs/sbert_rq1_n50_baseline_by_item.csv
    outputs/sbert_rq1_n50_baseline_by_task.csv
    outputs/rq1_n50_baseline_significance_tests.csv
    outputs/rq1_n50_baseline_tukey.csv
"""

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from statsmodels.stats.multicomp import pairwise_tukeyhsd


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GENERATION_FILES = [
    ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
    ROOT / "outputs" / "rq1_formal_original_generations_n50_math_reasoning.csv",
    ROOT / "outputs" / "rq1_formal_original_generations_n50_code_generation.csv",
    ROOT / "outputs" / "rq1_formal_original_generations_n50_open_ended_writing.csv",
]

BY_ITEM = ROOT / "outputs" / "sbert_rq1_n50_baseline_by_item.csv"
BY_TASK = ROOT / "outputs" / "sbert_rq1_n50_baseline_by_task.csv"
TESTS = ROOT / "outputs" / "rq1_n50_baseline_significance_tests.csv"
TUKEY = ROOT / "outputs" / "rq1_n50_baseline_tukey.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
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
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


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
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        return float(cos_sim(emb_a, emb_b)[0][0])

    def within_similarity(self, outputs: list[str]) -> float:
        return mean(
            [
                self.similarity(output_a, output_b)
                for output_a, output_b in combinations(outputs, 2)
            ]
        )


def load_rows() -> list[dict[str, str]]:
    rows = []
    for path in GENERATION_FILES:
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")
        file_rows = read_csv(path)
        print(f"Loaded {len(file_rows)} rows from {path.name}")
        rows.extend(file_rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    empty_outputs = 0
    for row in rows:
        counts[(row["task_type"], row["item_id"])] += 1
        if not row.get("output_text", "").strip():
            empty_outputs += 1

    bad = [(key, value) for key, value in counts.items() if value != 5]
    if empty_outputs:
        raise SystemExit(f"Found {empty_outputs} empty output_text rows.")
    if bad:
        examples = ", ".join(f"{key}={value}" for key, value in bad[:10])
        raise SystemExit(f"Some prompts do not have exactly 5 outputs: {examples}")
    print(f"Validated {len(counts)} prompts with 5 outputs each.")


def calculate_baseline(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    dataset_by_item: dict[tuple[str, str], str] = {}

    for row in rows:
        key = (row["item_id"], row["task_type"])
        grouped[key].append(row["output_text"])
        dataset_by_item[key] = row["dataset_name"]

    sim_model = SimilarityModel()
    item_rows = []
    task_drifts: dict[str, list[float]] = defaultdict(list)
    task_similarities: dict[str, list[float]] = defaultdict(list)

    for (item_id, task_type), outputs in sorted(grouped.items()):
        avg_similarity = sim_model.within_similarity(outputs)
        drift = 1 - avg_similarity
        n_outputs = len(outputs)
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "dataset_name": dataset_by_item[(item_id, task_type)],
                "n_outputs": n_outputs,
                "n_pairs": n_outputs * (n_outputs - 1) // 2,
                "mean_within_prompt_similarity": round(avg_similarity, 6),
                "sampling_noise_drift": round(drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )
        task_drifts[task_type].append(drift)
        task_similarities[task_type].append(avg_similarity)

    task_rows = []
    for task_type in sorted(task_drifts):
        task_rows.append(
            {
                "task_type": task_type,
                "n_items": len(task_drifts[task_type]),
                "mean_within_prompt_similarity": round(
                    mean(task_similarities[task_type]), 6
                ),
                "mean_sampling_noise_drift": round(mean(task_drifts[task_type]), 6),
                "std_sampling_noise_drift": round(
                    sample_std(task_drifts[task_type]), 6
                ),
                "similarity_metric": MODEL_NAME,
            }
        )

    write_csv(
        BY_ITEM,
        item_rows,
        [
            "item_id",
            "task_type",
            "dataset_name",
            "n_outputs",
            "n_pairs",
            "mean_within_prompt_similarity",
            "sampling_noise_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        BY_TASK,
        task_rows,
        [
            "task_type",
            "n_items",
            "mean_within_prompt_similarity",
            "mean_sampling_noise_drift",
            "std_sampling_noise_drift",
            "similarity_metric",
        ],
    )


def run_statistics() -> None:
    df = pd.read_csv(BY_ITEM)
    df["sampling_noise_drift"] = df["sampling_noise_drift"].astype(float)
    groups = [
        group["sampling_noise_drift"].to_numpy()
        for _, group in df.groupby("task_type")
    ]

    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    levene = stats.levene(*groups, center="median")

    write_csv(
        TESTS,
        [
            {
                "test": "one_way_anova",
                "statistic": round(float(anova.statistic), 6),
                "p_value": round(float(anova.pvalue), 6),
                "interpretation": "tests mean baseline drift differences across task types",
            },
            {
                "test": "kruskal_wallis",
                "statistic": round(float(kruskal.statistic), 6),
                "p_value": round(float(kruskal.pvalue), 6),
                "interpretation": "nonparametric check for baseline drift differences across task types",
            },
            {
                "test": "levene_median",
                "statistic": round(float(levene.statistic), 6),
                "p_value": round(float(levene.pvalue), 6),
                "interpretation": "checks equality of variance across task types",
            },
        ],
    )

    tukey = pairwise_tukeyhsd(
        endog=df["sampling_noise_drift"],
        groups=df["task_type"],
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
                "p_adj": round(float(p_adj), 6),
                "ci_lower": round(float(lower), 6),
                "ci_upper": round(float(upper), 6),
                "reject_alpha_0_05": bool(reject),
            }
        )
    write_csv(TUKEY, tukey_rows)


def main() -> None:
    rows = load_rows()
    validate_rows(rows)
    calculate_baseline(rows)
    run_statistics()
    print(f"Wrote {BY_ITEM}")
    print(f"Wrote {BY_TASK}")
    print(f"Wrote {TESTS}")
    print(f"Wrote {TUKEY}")


if __name__ == "__main__":
    main()
