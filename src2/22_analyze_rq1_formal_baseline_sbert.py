"""Calculate the formal RQ1 sampling-noise baseline with Sentence-BERT.

Input:
    outputs/rq1_formal_original_generations.csv

Outputs:
    outputs/sbert_rq1_formal_baseline_by_item.csv
    outputs/sbert_rq1_formal_baseline_by_task.csv

This script does not call the OpenAI API.
"""

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GENERATIONS = ROOT / "outputs" / "rq1_formal_original_generations.csv"
BY_ITEM = ROOT / "outputs" / "sbert_rq1_formal_baseline_by_item.csv"
BY_TASK = ROOT / "outputs" / "sbert_rq1_formal_baseline_by_task.csv"


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


def main() -> None:
    if not GENERATIONS.exists():
        raise SystemExit(f"Missing input file: {GENERATIONS}")

    rows = read_csv(GENERATIONS)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in rows:
        grouped[(row["item_id"], row["task_type"])].append(row["output_text"])

    sim_model = SimilarityModel()
    item_rows: list[dict] = []
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

    print(f"Wrote item-level baseline to {BY_ITEM}")
    print(f"Wrote task-level baseline to {BY_TASK}")


if __name__ == "__main__":
    main()
