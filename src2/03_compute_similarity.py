"""Compute pilot similarity metrics for RQ1.

The project proposal eventually uses Sentence-BERT. This pilot version uses a
small bag-of-words cosine similarity so the experiment structure can be tested
without installing model dependencies.
"""

import csv
import math
import re
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "outputs" / "generations.csv"
SIMILARITY_RESULTS = ROOT / "outputs" / "similarity_results.csv"
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

    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[token] * vec_b[token] for token in shared)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    return dot / (norm_a * norm_b)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pairwise_within_similarity(texts: list[str]) -> float:
    pairs = list(combinations(texts, 2))
    return mean([cosine_similarity(a, b) for a, b in pairs])


def cross_similarity(original_texts: list[str], perturbed_texts: list[str]) -> float:
    pairs = list(product(original_texts, perturbed_texts))
    return mean([cosine_similarity(a, b) for a, b in pairs])


def main() -> None:
    rows = read_csv(GENERATIONS)
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    task_by_item: dict[str, str] = {}
    for row in rows:
        key = (row["item_id"], row["prompt_version"], row["perturbation_type"])
        grouped[key].append(row["output_text"])
        task_by_item[row["item_id"]] = row["task_type"]

    item_noise: dict[str, float] = {}
    for item_id in task_by_item:
        original_texts = grouped[(item_id, "original", "none")]
        item_noise[item_id] = pairwise_within_similarity(original_texts)

    task_noise_values: dict[str, list[float]] = defaultdict(list)
    for item_id, noise in item_noise.items():
        task_noise_values[task_by_item[item_id]].append(noise)
    task_noise = {
        task_type: mean(values) for task_type, values in task_noise_values.items()
    }

    result_rows: list[dict[str, str | float]] = []
    for (item_id, prompt_version, perturbation_type), perturbed_texts in grouped.items():
        if prompt_version != "perturbed":
            continue

        task_type = task_by_item[item_id]
        original_texts = grouped[(item_id, "original", "none")]
        perturbation_similarity = cross_similarity(original_texts, perturbed_texts)
        item_baseline = item_noise[item_id]
        task_baseline = task_noise[task_type]

        result_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "item_noise_baseline": round(item_baseline, 6),
                "task_noise_baseline": round(task_baseline, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "noise_corrected_drift_item": round(
                    item_baseline - perturbation_similarity, 6
                ),
                "noise_corrected_drift_task": round(
                    task_baseline - perturbation_similarity, 6
                ),
            }
        )

    result_rows.sort(
        key=lambda row: (
            str(row["task_type"]),
            str(row["item_id"]),
            str(row["perturbation_type"]),
        )
    )

    with SIMILARITY_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "item_id",
                "task_type",
                "perturbation_type",
                "item_noise_baseline",
                "task_noise_baseline",
                "perturbation_similarity",
                "noise_corrected_drift_item",
                "noise_corrected_drift_task",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"Wrote {len(result_rows)} similarity rows to {SIMILARITY_RESULTS}")


if __name__ == "__main__":
    main()
