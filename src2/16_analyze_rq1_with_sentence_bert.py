"""Recompute RQ1 analyses with Sentence-BERT similarity.

This script reads existing LLM output CSV files and writes new result files
prefixed with sbert_. It does not call the LLM API.
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

RQ1A_GENERATIONS = ROOT / "outputs" / "rq1_generations.csv"
RQ1B_GENERATIONS = ROOT / "outputs" / "rq1b_pilot_perturbed_generations.csv"
CALIBRATION_GENERATIONS = ROOT / "outputs" / "rq1_calibration_generations.csv"

SBERT_RQ1A_ITEM = ROOT / "outputs" / "sbert_rq1a_noise_by_item.csv"
SBERT_RQ1A_TASK = ROOT / "outputs" / "sbert_rq1a_noise_by_task.csv"

SBERT_RQ1B_ITEM = ROOT / "outputs" / "sbert_rq1b_perturbation_effects_by_item.csv"
SBERT_RQ1B_SUMMARY = ROOT / "outputs" / "sbert_rq1b_perturbation_summary.csv"
SBERT_RQ1B_HEATMAP = ROOT / "outputs" / "sbert_rq1b_heatmap_noise_corrected_drift.csv"

SBERT_CALIBRATION_ITEM = ROOT / "outputs" / "sbert_rq1_baseline_stability_by_item.csv"
SBERT_CALIBRATION_TASK = ROOT / "outputs" / "sbert_rq1_baseline_stability_by_task.csv"

N_VALUES = [3, 5, 7, 10]
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
        self.cache: dict[str, object] = {}

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


def analyze_rq1a(sim_model: SimilarityModel) -> dict[str, float]:
    rows = read_csv(RQ1A_GENERATIONS)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in rows:
        grouped[(row["item_id"], row["task_type"])].append(row["output_text"])

    item_rows: list[dict] = []
    task_drifts: dict[str, list[float]] = defaultdict(list)
    task_similarities: dict[str, list[float]] = defaultdict(list)
    baseline_by_item: dict[str, float] = {}

    for (item_id, task_type), outputs in sorted(grouped.items()):
        avg_similarity = sim_model.within_similarity(outputs)
        drift = 1 - avg_similarity
        baseline_by_item[item_id] = avg_similarity
        task_drifts[task_type].append(drift)
        task_similarities[task_type].append(avg_similarity)
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "n_outputs": len(outputs),
                "n_pairs": len(outputs) * (len(outputs) - 1) // 2,
                "mean_within_prompt_similarity": round(avg_similarity, 6),
                "sampling_noise_drift": round(drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )

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
        SBERT_RQ1A_ITEM,
        item_rows,
        [
            "item_id",
            "task_type",
            "n_outputs",
            "n_pairs",
            "mean_within_prompt_similarity",
            "sampling_noise_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        SBERT_RQ1A_TASK,
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
    return baseline_by_item


def analyze_rq1b(sim_model: SimilarityModel) -> None:
    original_rows = read_csv(RQ1A_GENERATIONS)
    perturbed_rows = read_csv(RQ1B_GENERATIONS)

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

    detail_rows: list[dict] = []
    for (item_id, perturbation_type), perturbed_outputs in sorted(
        perturbed_by_group.items()
    ):
        original_outputs = original_by_item[item_id]
        baseline_similarity = baseline_by_item[item_id]
        perturbation_similarity = sim_model.cross_similarity(
            original_outputs, perturbed_outputs
        )
        drift = baseline_similarity - perturbation_similarity
        detail_rows.append(
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
    for row in detail_rows:
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
        SBERT_RQ1B_ITEM,
        detail_rows,
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
        SBERT_RQ1B_SUMMARY,
        summary_rows,
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_noise_corrected_drift",
            "similarity_metric",
        ],
    )
    write_csv(SBERT_RQ1B_HEATMAP, heatmap_rows, ["perturbation_type", *task_types])


def analyze_calibration(sim_model: SimilarityModel) -> None:
    rows = read_csv(CALIBRATION_GENERATIONS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[(row["item_id"], row["task_type"])].append(row)

    item_rows: list[dict] = []
    task_values: dict[tuple[str, int], list[float]] = defaultdict(list)

    for (item_id, task_type), group_rows in sorted(grouped.items()):
        sorted_rows = sorted(group_rows, key=lambda row: int(row["sample_id"]))
        outputs = [row["output_text"] for row in sorted_rows]
        previous_drift: float | None = None

        for n in N_VALUES:
            avg_similarity = sim_model.within_similarity(outputs[:n])
            drift = 1 - avg_similarity
            change = "" if previous_drift is None else round(abs(drift - previous_drift), 6)
            previous_drift = drift
            task_values[(task_type, n)].append(drift)
            item_rows.append(
                {
                    "item_id": item_id,
                    "task_type": task_type,
                    "n_outputs": n,
                    "n_pairs": n * (n - 1) // 2,
                    "sampling_noise_drift": round(drift, 6),
                    "abs_change_from_previous_n": change,
                    "similarity_metric": MODEL_NAME,
                }
            )

    task_rows: list[dict] = []
    for (task_type, n), values in sorted(task_values.items()):
        task_rows.append(
            {
                "task_type": task_type,
                "n_outputs": n,
                "n_items": len(values),
                "mean_sampling_noise_drift": round(mean(values), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    write_csv(
        SBERT_CALIBRATION_ITEM,
        item_rows,
        [
            "item_id",
            "task_type",
            "n_outputs",
            "n_pairs",
            "sampling_noise_drift",
            "abs_change_from_previous_n",
            "similarity_metric",
        ],
    )
    write_csv(
        SBERT_CALIBRATION_TASK,
        task_rows,
        [
            "task_type",
            "n_outputs",
            "n_items",
            "mean_sampling_noise_drift",
            "similarity_metric",
        ],
    )


def main() -> None:
    sim_model = SimilarityModel()
    analyze_rq1a(sim_model)
    analyze_rq1b(sim_model)
    analyze_calibration(sim_model)

    print("Wrote Sentence-BERT RQ1 result files:")
    for path in [
        SBERT_RQ1A_ITEM,
        SBERT_RQ1A_TASK,
        SBERT_RQ1B_ITEM,
        SBERT_RQ1B_SUMMARY,
        SBERT_RQ1B_HEATMAP,
        SBERT_CALIBRATION_ITEM,
        SBERT_CALIBRATION_TASK,
    ]:
        print(path)


if __name__ == "__main__":
    main()
