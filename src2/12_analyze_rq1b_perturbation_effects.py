"""Analyze RQ1b pilot perturbation effects.

This compares original-prompt outputs from RQ1a with perturbed-prompt outputs
from RQ1b, then applies a noise-baseline correction.
"""

import csv
import math
import re
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_GENERATIONS = ROOT / "outputs" / "rq1_generations.csv"
PERTURBED_GENERATIONS = ROOT / "outputs" / "rq1b_pilot_perturbed_generations.csv"
DETAIL_RESULTS = ROOT / "outputs" / "rq1b_pilot_perturbation_effects_by_item.csv"
SUMMARY_RESULTS = ROOT / "outputs" / "rq1b_pilot_perturbation_summary.csv"
HEATMAP_RESULTS = ROOT / "outputs" / "rq1b_pilot_heatmap_noise_corrected_drift.csv"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
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


def within_similarity(outputs: list[str]) -> float:
    return mean(
        [
            cosine_similarity(output_a, output_b)
            for output_a, output_b in combinations(outputs, 2)
        ]
    )


def cross_similarity(original_outputs: list[str], perturbed_outputs: list[str]) -> float:
    return mean(
        [
            cosine_similarity(original_output, perturbed_output)
            for original_output, perturbed_output in product(
                original_outputs, perturbed_outputs
            )
        ]
    )


def main() -> None:
    original_rows = read_csv(ORIGINAL_GENERATIONS)
    perturbed_rows = read_csv(PERTURBED_GENERATIONS)

    original_by_item: dict[str, list[str]] = defaultdict(list)
    task_by_item: dict[str, str] = {}
    for row in original_rows:
        original_by_item[row["item_id"]].append(row["output_text"])
        task_by_item[row["item_id"]] = row["task_type"]

    baseline_by_item = {
        item_id: within_similarity(outputs)
        for item_id, outputs in original_by_item.items()
    }

    perturbed_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(
            row["output_text"]
        )

    detail_rows: list[dict[str, str | int | float]] = []
    for (item_id, perturbation_type), perturbed_outputs in sorted(
        perturbed_by_group.items()
    ):
        original_outputs = original_by_item[item_id]
        baseline_similarity = baseline_by_item[item_id]
        perturbation_similarity = cross_similarity(
            original_outputs, perturbed_outputs
        )
        noise_corrected_drift = baseline_similarity - perturbation_similarity

        detail_rows.append(
            {
                "item_id": item_id,
                "task_type": task_by_item[item_id],
                "perturbation_type": perturbation_type,
                "n_original_outputs": len(original_outputs),
                "n_perturbed_outputs": len(perturbed_outputs),
                "baseline_similarity": round(baseline_similarity, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "noise_corrected_drift": round(noise_corrected_drift, 6),
            }
        )

    with DETAIL_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "item_id",
                "task_type",
                "perturbation_type",
                "n_original_outputs",
                "n_perturbed_outputs",
                "baseline_similarity",
                "perturbation_similarity",
                "noise_corrected_drift",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["task_type"]), str(row["perturbation_type"]))].append(
            float(row["noise_corrected_drift"])
        )

    summary_rows: list[dict[str, str | int | float]] = []
    for (task_type, perturbation_type), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(values),
                "mean_noise_corrected_drift": round(mean(values), 6),
            }
        )

    with SUMMARY_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_type",
                "perturbation_type",
                "n_items",
                "mean_noise_corrected_drift",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    task_types = sorted({str(row["task_type"]) for row in summary_rows})
    lookup = {
        (str(row["task_type"]), str(row["perturbation_type"])): row[
            "mean_noise_corrected_drift"
        ]
        for row in summary_rows
    }
    with HEATMAP_RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["perturbation_type", *task_types])
        for perturbation_type in PERTURBATION_ORDER:
            writer.writerow(
                [
                    perturbation_type,
                    *[
                        lookup.get((task_type, perturbation_type), "")
                        for task_type in task_types
                    ],
                ]
            )

    print(f"Wrote item-level effects to {DETAIL_RESULTS}")
    print(f"Wrote summary to {SUMMARY_RESULTS}")
    print(f"Wrote heatmap table to {HEATMAP_RESULTS}")
    print()
    print("Largest RQ1b pilot noise-corrected drift values:")
    for row in sorted(
        summary_rows,
        key=lambda value: float(value["mean_noise_corrected_drift"]),
        reverse=True,
    )[:10]:
        print(
            f"{row['task_type']} / {row['perturbation_type']}: "
            f"{row['mean_noise_corrected_drift']}"
        )


if __name__ == "__main__":
    main()
