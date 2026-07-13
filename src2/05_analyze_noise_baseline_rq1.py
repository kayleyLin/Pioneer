"""Analyze RQ1 as sampling-noise baseline only.

This script does not use perturbed prompts. It measures how similar repeated
outputs are when the same original prompt is sent to the same model multiple
times.
"""

import csv
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "outputs" / "generations.csv"
ITEM_RESULTS = ROOT / "outputs" / "rq1_noise_baseline_by_item.csv"
TASK_RESULTS = ROOT / "outputs" / "rq1_noise_baseline_by_task.csv"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def vectorize(text: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(text.lower()))


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


def main() -> None:
    rows = [
        row
        for row in read_csv(GENERATIONS)
        if row["prompt_version"] == "original"
        and row["perturbation_type"] == "none"
    ]

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        grouped[(row["item_id"], row["task_type"])].append(row["output_text"])

    item_rows: list[dict[str, str | int | float]] = []
    task_values: dict[str, list[float]] = defaultdict(list)

    for (item_id, task_type), outputs in sorted(grouped.items()):
        similarities = [
            cosine_similarity(output_a, output_b)
            for output_a, output_b in combinations(outputs, 2)
        ]
        avg_similarity = mean(similarities)
        noise_drift = 1 - avg_similarity
        task_values[task_type].append(noise_drift)

        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "n_outputs": len(outputs),
                "n_pairs": len(similarities),
                "mean_within_prompt_similarity": round(avg_similarity, 6),
                "sampling_noise_drift": round(noise_drift, 6),
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
            ],
        )
        writer.writeheader()
        writer.writerows(item_rows)

    task_rows = [
        {
            "task_type": task_type,
            "n_items": len(values),
            "mean_sampling_noise_drift": round(mean(values), 6),
        }
        for task_type, values in sorted(task_values.items())
    ]

    with TASK_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task_type", "n_items", "mean_sampling_noise_drift"],
        )
        writer.writeheader()
        writer.writerows(task_rows)

    print(f"Wrote item-level RQ1 baseline to {ITEM_RESULTS}")
    print(f"Wrote task-level RQ1 baseline to {TASK_RESULTS}")
    print()
    print("RQ1 task-level sampling noise:")
    for row in task_rows:
        print(f"{row['task_type']}: {row['mean_sampling_noise_drift']}")


if __name__ == "__main__":
    main()
