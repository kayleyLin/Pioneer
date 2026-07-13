"""Analyze RQ1 pilot results."""

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_RESULTS = ROOT / "outputs" / "similarity_results.csv"
SUMMARY_BY_TASK = ROOT / "outputs" / "rq1_summary_by_task_perturbation.csv"
HEATMAP = ROOT / "outputs" / "rq1_heatmap_noise_corrected_drift.csv"


PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting",
    "context_injection",
    "surface_noise",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    rows = read_csv(SIMILARITY_RESULTS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[(row["task_type"], row["perturbation_type"])].append(row)

    summary_rows: list[dict[str, str | int | float]] = []
    for (task_type, perturbation_type), group_rows in sorted(grouped.items()):
        item_drifts = [
            float(row["noise_corrected_drift_item"]) for row in group_rows
        ]
        task_drifts = [
            float(row["noise_corrected_drift_task"]) for row in group_rows
        ]
        perturbation_similarities = [
            float(row["perturbation_similarity"]) for row in group_rows
        ]
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(group_rows),
                "mean_perturbation_similarity": round(
                    mean(perturbation_similarities), 6
                ),
                "mean_noise_corrected_drift_item": round(mean(item_drifts), 6),
                "mean_noise_corrected_drift_task": round(mean(task_drifts), 6),
            }
        )

    with SUMMARY_BY_TASK.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_type",
                "perturbation_type",
                "n_items",
                "mean_perturbation_similarity",
                "mean_noise_corrected_drift_item",
                "mean_noise_corrected_drift_task",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    task_types = sorted({row["task_type"] for row in rows})
    drift_lookup = {
        (str(row["task_type"]), str(row["perturbation_type"])): row[
            "mean_noise_corrected_drift_task"
        ]
        for row in summary_rows
    }

    with HEATMAP.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["perturbation_type", *task_types])
        for perturbation_type in PERTURBATION_ORDER:
            writer.writerow(
                [
                    perturbation_type,
                    *[
                        drift_lookup.get((task_type, perturbation_type), "")
                        for task_type in task_types
                    ],
                ]
            )

    print(f"Wrote summary to {SUMMARY_BY_TASK}")
    print(f"Wrote heatmap table to {HEATMAP}")
    print()
    print("Largest task-level noise-corrected drift values:")
    for row in sorted(
        summary_rows,
        key=lambda value: float(value["mean_noise_corrected_drift_task"]),
        reverse=True,
    )[:8]:
        print(
            f"{row['task_type']} / {row['perturbation_type']}: "
            f"{row['mean_noise_corrected_drift_task']}"
        )


if __name__ == "__main__":
    main()
