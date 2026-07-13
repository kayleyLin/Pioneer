"""Analyze RQ1 real LLM generations.

Input:
    outputs/rq1_generations.csv

Outputs:
    outputs/rq1_real_noise_by_item.csv
    outputs/rq1_real_noise_by_task.csv
"""

import csv
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "outputs" / "rq1_generations.csv"
ITEM_RESULTS = ROOT / "outputs" / "rq1_real_noise_by_item.csv"
TASK_RESULTS = ROOT / "outputs" / "rq1_real_noise_by_task.csv"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(text_a: str, text_b: str) -> float:
    vec_a = vectorize(text_a)
    vec_b = vectorize(text_b)
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(vec_a[token] * vec_b[token] for token in set(vec_a) & set(vec_b))
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    return dot / (norm_a * norm_b)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def main() -> None:
    rows = read_csv(GENERATIONS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[(row["item_id"], row["task_type"])].append(row)

    item_rows: list[dict[str, str | int | float]] = []
    task_drift_values: dict[str, list[float]] = defaultdict(list)
    task_similarity_values: dict[str, list[float]] = defaultdict(list)

    for (item_id, task_type), group_rows in sorted(grouped.items()):
        outputs = [row["output_text"] for row in group_rows]
        pair_similarities = [
            cosine_similarity(output_a, output_b)
            for output_a, output_b in combinations(outputs, 2)
        ]
        avg_similarity = mean(pair_similarities)
        noise_drift = 1 - avg_similarity

        task_similarity_values[task_type].append(avg_similarity)
        task_drift_values[task_type].append(noise_drift)

        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "n_outputs": len(outputs),
                "n_pairs": len(pair_similarities),
                "mean_within_prompt_similarity": round(avg_similarity, 6),
                "sampling_noise_drift": round(noise_drift, 6),
                "pairwise_similarities": ";".join(
                    str(round(value, 6)) for value in pair_similarities
                ),
            }
        )

    with ITEM_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "item_id",
                "task_type",
                "n_outputs",
                "n_pairs",
                "mean_within_prompt_similarity",
                "sampling_noise_drift",
                "pairwise_similarities",
            ],
        )
        writer.writeheader()
        writer.writerows(item_rows)

    task_rows: list[dict[str, str | int | float]] = []
    for task_type in sorted(task_drift_values):
        drifts = task_drift_values[task_type]
        similarities = task_similarity_values[task_type]
        task_rows.append(
            {
                "task_type": task_type,
                "n_items": len(drifts),
                "mean_within_prompt_similarity": round(mean(similarities), 6),
                "mean_sampling_noise_drift": round(mean(drifts), 6),
                "std_sampling_noise_drift": round(sample_std(drifts), 6),
            }
        )

    with TASK_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_type",
                "n_items",
                "mean_within_prompt_similarity",
                "mean_sampling_noise_drift",
                "std_sampling_noise_drift",
            ],
        )
        writer.writeheader()
        writer.writerows(task_rows)

    print(f"Wrote item-level results to {ITEM_RESULTS}")
    print(f"Wrote task-level results to {TASK_RESULTS}")
    print()
    print("RQ1 task-level results:")
    for row in sorted(
        task_rows,
        key=lambda value: float(value["mean_sampling_noise_drift"]),
        reverse=True,
    ):
        print(
            f"{row['task_type']}: drift={row['mean_sampling_noise_drift']}, "
            f"similarity={row['mean_within_prompt_similarity']}"
        )


if __name__ == "__main__":
    main()
