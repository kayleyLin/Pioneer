"""Analyze baseline stability for different repeated-generation counts."""

import csv
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "outputs" / "rq1_calibration_generations.csv"
BY_ITEM = ROOT / "outputs" / "rq1_baseline_stability_by_item.csv"
BY_TASK = ROOT / "outputs" / "rq1_baseline_stability_by_task.csv"
N_VALUES = [3, 5, 7, 10]
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


def noise_drift(outputs: list[str]) -> float:
    similarities = [
        cosine_similarity(output_a, output_b)
        for output_a, output_b in combinations(outputs, 2)
    ]
    return 1 - mean(similarities)


def main() -> None:
    rows = read_csv(GENERATIONS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[(row["item_id"], row["task_type"])].append(row)

    item_rows: list[dict[str, str | int | float]] = []
    task_values: dict[tuple[str, int], list[float]] = defaultdict(list)

    for (item_id, task_type), group_rows in sorted(grouped.items()):
        sorted_rows = sorted(group_rows, key=lambda row: int(row["sample_id"]))
        outputs = [row["output_text"] for row in sorted_rows]
        if len(outputs) < max(N_VALUES):
            raise ValueError(f"{item_id} has only {len(outputs)} outputs")

        previous_drift: float | None = None
        for n in N_VALUES:
            drift = noise_drift(outputs[:n])
            change_from_previous = (
                "" if previous_drift is None else round(abs(drift - previous_drift), 6)
            )
            previous_drift = drift
            task_values[(task_type, n)].append(drift)
            item_rows.append(
                {
                    "item_id": item_id,
                    "task_type": task_type,
                    "n_outputs": n,
                    "n_pairs": n * (n - 1) // 2,
                    "sampling_noise_drift": round(drift, 6),
                    "abs_change_from_previous_n": change_from_previous,
                }
            )

    with BY_ITEM.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "item_id",
                "task_type",
                "n_outputs",
                "n_pairs",
                "sampling_noise_drift",
                "abs_change_from_previous_n",
            ],
        )
        writer.writeheader()
        writer.writerows(item_rows)

    task_rows: list[dict[str, str | int | float]] = []
    for (task_type, n), values in sorted(task_values.items()):
        task_rows.append(
            {
                "task_type": task_type,
                "n_outputs": n,
                "n_items": len(values),
                "mean_sampling_noise_drift": round(mean(values), 6),
            }
        )

    with BY_TASK.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_type",
                "n_outputs",
                "n_items",
                "mean_sampling_noise_drift",
            ],
        )
        writer.writeheader()
        writer.writerows(task_rows)

    print(f"Wrote item-level stability results to {BY_ITEM}")
    print(f"Wrote task-level stability results to {BY_TASK}")
    print()
    print("Task-level baseline drift by n:")
    for row in task_rows:
        print(
            f"{row['task_type']} n={row['n_outputs']}: "
            f"{row['mean_sampling_noise_drift']}"
        )


if __name__ == "__main__":
    main()
