"""Calculate formal RQ1 perturbation effects with Sentence-BERT.

Inputs:
    outputs/rq1_formal_original_generations.csv
    outputs/rq1_formal_perturbed_generations.csv

Outputs:
    outputs/sbert_rq1_formal_perturbation_effects_by_item.csv
    outputs/sbert_rq1_formal_perturbation_summary.csv
    outputs/sbert_rq1_formal_heatmap_noise_corrected_drift.csv

This script does not call the OpenAI API.
"""

import csv
import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ORIGINAL_GENERATIONS = ROOT / "outputs" / "rq1_formal_original_generations.csv"
PERTURBED_GENERATIONS = ROOT / "outputs" / "rq1_formal_perturbed_generations.csv"
BY_ITEM = ROOT / "outputs" / "sbert_rq1_formal_perturbation_effects_by_item.csv"
SUMMARY = ROOT / "outputs" / "sbert_rq1_formal_perturbation_summary.csv"
HEATMAP = ROOT / "outputs" / "sbert_rq1_formal_heatmap_noise_corrected_drift.csv"

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


def main() -> None:
    for path in [ORIGINAL_GENERATIONS, PERTURBED_GENERATIONS]:
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")

    original_rows = read_csv(ORIGINAL_GENERATIONS)
    perturbed_rows = read_csv(PERTURBED_GENERATIONS)
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
        drift = baseline_similarity - perturbation_similarity
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_by_item[item_id],
                "perturbation_type": perturbation_type,
                "n_original_outputs": len(original_outputs),
                "n_perturbed_outputs": len(perturbed_outputs),
                "baseline_similarity": round(baseline_similarity, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "noise_corrected_drift": round(drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in item_rows:
        grouped[(row["task_type"], row["perturbation_type"])].append(
            row["noise_corrected_drift"]
        )

    summary_rows: list[dict] = []
    for (task_type, perturbation_type), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(values),
                "mean_noise_corrected_drift": round(mean(values), 6),
                "std_noise_corrected_drift": round(sample_std(values), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    task_types = sorted({row["task_type"] for row in summary_rows})
    lookup = {
        (row["task_type"], row["perturbation_type"]): row[
            "mean_noise_corrected_drift"
        ]
        for row in summary_rows
    }
    heatmap_rows = []
    for perturbation_type in PERTURBATION_ORDER:
        row = {"perturbation_type": perturbation_type}
        for task_type in task_types:
            row[task_type] = lookup.get((task_type, perturbation_type), "")
        heatmap_rows.append(row)

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
    write_csv(HEATMAP, heatmap_rows, ["perturbation_type", *task_types])

    print(f"Wrote item-level perturbation effects to {BY_ITEM}")
    print(f"Wrote perturbation summary to {SUMMARY}")
    print(f"Wrote heatmap to {HEATMAP}")


if __name__ == "__main__":
    main()
