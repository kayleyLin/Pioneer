"""Analyze expanded RQ1 n=50 perturbation effects with Sentence-BERT (Qwen outputs).

This script does not call any LLM API.

Inputs:
    qwen/outputs/rq1_qwen_original_generations_n50_*.csv
    qwen/outputs/rq1_qwen_perturbed_generations_n50_*.csv

Outputs:
    qwen/outputs/sbert_rq1_n50_perturbation_effects_by_item.csv
    qwen/outputs/sbert_rq1_n50_perturbation_summary.csv
    qwen/outputs/sbert_rq1_n50_heatmap_noise_corrected_drift.csv
    qwen/outputs/sbert_rq1_n50_uncorrected_perturbation_summary.csv
    qwen/outputs/sbert_rq1_n50_uncorrected_heatmap_drift.csv
"""

import csv
import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "qwen"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ORIGINAL_FILES = [
    QWEN / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
    QWEN / "outputs" / "rq1_qwen_original_generations_n50_math_reasoning.csv",
    QWEN / "outputs" / "rq1_qwen_original_generations_n50_code_generation.csv",
    QWEN / "outputs" / "rq1_qwen_original_generations_n50_open_ended_writing.csv",
]
PERTURBED_FILES = [
    QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa.csv",
    QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_math_reasoning.csv",
    QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_code_generation.csv",
    QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_open_ended_writing.csv",
]

BY_ITEM = QWEN / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item.csv"
SUMMARY = QWEN / "outputs" / "sbert_rq1_n50_perturbation_summary.csv"
HEATMAP = QWEN / "outputs" / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv"
UNCORRECTED_SUMMARY = QWEN / "outputs" / "sbert_rq1_n50_uncorrected_perturbation_summary.csv"
UNCORRECTED_HEATMAP = QWEN / "outputs" / "sbert_rq1_n50_uncorrected_heatmap_drift.csv"

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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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

    def cross_similarity(
        self, original_outputs: list[str], perturbed_outputs: list[str]
    ) -> float:
        return mean(
            [
                self.similarity(original_output, perturbed_output)
                for original_output, perturbed_output in product(
                    original_outputs, perturbed_outputs
                )
            ]
        )


def load_all(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")
        file_rows = read_csv(path)
        print(f"Loaded {len(file_rows)} rows from {path.name}")
        rows.extend(file_rows)
    return rows


def validate_original_rows(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[row["item_id"]] += 1
    bad = {item_id: count for item_id, count in grouped.items() if count != 5}
    empty = sum(not row.get("output_text", "").strip() for row in rows)
    if empty:
        raise SystemExit(f"Found {empty} empty original output rows.")
    if bad:
        raise SystemExit(f"Original rows with non-5 samples: {list(bad.items())[:10]}")
    print(f"Validated {len(grouped)} original prompts with 5 outputs each.")


def validate_perturbed_rows(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(row["item_id"], row["perturbation_type"])] += 1
    bad = {key: count for key, count in grouped.items() if count != 5}
    empty = sum(not row.get("output_text", "").strip() for row in rows)
    if empty:
        raise SystemExit(f"Found {empty} empty perturbed output rows.")
    if bad:
        raise SystemExit(f"Perturbed rows with non-5 samples: {list(bad.items())[:10]}")
    print(f"Validated {len(grouped)} item-perturbation pairs with 5 outputs each.")


def heatmap_rows(summary_rows: list[dict], value_key: str) -> list[dict]:
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


def main() -> None:
    original_rows = load_all(ORIGINAL_FILES)
    perturbed_rows = load_all(PERTURBED_FILES)
    validate_original_rows(original_rows)
    validate_perturbed_rows(perturbed_rows)

    sim_model = SimilarityModel()

    original_by_item: dict[str, list[str]] = defaultdict(list)
    task_by_item: dict[str, str] = {}
    for row in original_rows:
        original_by_item[row["item_id"]].append(row["output_text"])
        task_by_item[row["item_id"]] = row["task_type"]

    baseline_by_item = {
        item_id: sim_model.within_similarity(outputs)
        for item_id, outputs in original_by_item.items()
    }

    perturbed_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(
            row["output_text"]
        )

    item_rows: list[dict] = []
    for (item_id, perturbation_type), perturbed_outputs in sorted(
        perturbed_by_group.items()
    ):
        original_outputs = original_by_item[item_id]
        baseline_similarity = baseline_by_item[item_id]
        perturbation_similarity = sim_model.cross_similarity(
            original_outputs, perturbed_outputs
        )
        uncorrected_drift = 1 - perturbation_similarity
        noise_corrected_drift = baseline_similarity - perturbation_similarity
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_by_item[item_id],
                "perturbation_type": perturbation_type,
                "n_original_outputs": len(original_outputs),
                "n_perturbed_outputs": len(perturbed_outputs),
                "baseline_similarity": round(baseline_similarity, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "uncorrected_drift": round(uncorrected_drift, 6),
                "noise_corrected_drift": round(noise_corrected_drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in item_rows:
        grouped[(row["task_type"], row["perturbation_type"])].append(row)

    summary_rows: list[dict] = []
    uncorrected_rows: list[dict] = []
    for (task_type, perturbation_type), rows in sorted(grouped.items()):
        corrected_values = [float(row["noise_corrected_drift"]) for row in rows]
        uncorrected_values = [float(row["uncorrected_drift"]) for row in rows]
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(rows),
                "mean_noise_corrected_drift": round(mean(corrected_values), 6),
                "std_noise_corrected_drift": round(sample_std(corrected_values), 6),
                "similarity_metric": MODEL_NAME,
            }
        )
        uncorrected_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(rows),
                "mean_uncorrected_drift": round(mean(uncorrected_values), 6),
                "std_uncorrected_drift": round(sample_std(uncorrected_values), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    task_types = sorted({row["task_type"] for row in summary_rows})
    write_csv(
        BY_ITEM,
        item_rows,
        [
            "item_id",
            "task_type",
            "perturbation_type",
            "n_original_outputs",
            "n_perturbed_outputs",
            "baseline_similarity",
            "perturbation_similarity",
            "uncorrected_drift",
            "noise_corrected_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        SUMMARY,
        summary_rows,
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_noise_corrected_drift",
            "std_noise_corrected_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        HEATMAP,
        heatmap_rows(summary_rows, "mean_noise_corrected_drift"),
        ["perturbation_type", *task_types],
    )
    write_csv(
        UNCORRECTED_SUMMARY,
        uncorrected_rows,
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_uncorrected_drift",
            "std_uncorrected_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        UNCORRECTED_HEATMAP,
        heatmap_rows(uncorrected_rows, "mean_uncorrected_drift"),
        ["perturbation_type", *task_types],
    )

    print(f"Wrote {BY_ITEM}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {HEATMAP}")
    print(f"Wrote {UNCORRECTED_SUMMARY}")
    print(f"Wrote {UNCORRECTED_HEATMAP}")


if __name__ == "__main__":
    main()
