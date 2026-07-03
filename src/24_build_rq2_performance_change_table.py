"""Build RQ2 item-level performance-change tables.

This script combines original and perturbed RQ2 performance labels into one row
per item and perturbation type. It does not require semantic-drift results, so it
can run as soon as original and perturbed correctness/performance files exist.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "rq2" / "outputs" / "rq2_original_correctness_by_generation.csv"
PERTURBED = ROOT / "rq2" / "outputs" / "rq2_formal_available_perturbed_performance_by_generation.csv"
OUT_BY_ITEM = ROOT / "rq2" / "outputs" / "rq2_formal_available_performance_change_by_item.csv"
OUT_SUMMARY = ROOT / "rq2" / "outputs" / "rq2_formal_available_performance_change_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score(row: dict[str, str]) -> float:
    value = row.get("performance_score", "")
    if value == "":
        raise ValueError(f"Missing performance_score for {row.get('item_id')}")
    return float(value)


def mean_score(rows: list[dict[str, str]]) -> float:
    return mean(score(row) for row in rows)


def safe_pdr(original_score: float, perturbed_score: float) -> str:
    if original_score == 0:
        return ""
    return f"{(original_score - perturbed_score) / original_score:.6f}"


def main() -> None:
    original_rows = read_csv(ORIGINAL)
    perturbed_rows = read_csv(PERTURBED)

    original_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in original_rows:
        original_by_item[row["item_id"]].append(row)

    perturbed_by_item: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_item[(row["item_id"], row["perturbation_type"])].append(row)

    item_rows: list[dict[str, str]] = []
    for (item_id, perturbation_type), rows in sorted(perturbed_by_item.items()):
        if item_id not in original_by_item:
            continue
        original_score = mean_score(original_by_item[item_id])
        perturbed_score = mean_score(rows)
        absolute_change = original_score - perturbed_score
        task_type = rows[0]["task_type"]
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_original_outputs": str(len(original_by_item[item_id])),
                "n_perturbed_outputs": str(len(rows)),
                "original_performance": f"{original_score:.6f}",
                "perturbed_performance": f"{perturbed_score:.6f}",
                "absolute_performance_change": f"{absolute_change:.6f}",
                "pdr": safe_pdr(original_score, perturbed_score),
                "performance_dropped": "true" if perturbed_score < original_score else "false",
            }
        )

    write_csv(
        OUT_BY_ITEM,
        item_rows,
        [
            "item_id",
            "task_type",
            "perturbation_type",
            "n_original_outputs",
            "n_perturbed_outputs",
            "original_performance",
            "perturbed_performance",
            "absolute_performance_change",
            "pdr",
            "performance_dropped",
        ],
    )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in item_rows:
        grouped[(row["task_type"], row["perturbation_type"])].append(row)

    summary_rows: list[dict[str, str]] = []
    for (task_type, perturbation_type), rows in sorted(grouped.items()):
        changes = [float(row["absolute_performance_change"]) for row in rows]
        pdr_values = [float(row["pdr"]) for row in rows if row["pdr"] != ""]
        dropped = sum(row["performance_dropped"] == "true" for row in rows)
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": str(len(rows)),
                "mean_original_performance": f"{mean(float(row['original_performance']) for row in rows):.6f}",
                "mean_perturbed_performance": f"{mean(float(row['perturbed_performance']) for row in rows):.6f}",
                "mean_absolute_performance_change": f"{mean(changes):.6f}",
                "mean_pdr": "" if not pdr_values else f"{mean(pdr_values):.6f}",
                "n_items_with_performance_drop": str(dropped),
            }
        )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_original_performance",
            "mean_perturbed_performance",
            "mean_absolute_performance_change",
            "mean_pdr",
            "n_items_with_performance_drop",
        ],
    )

    print(f"Wrote {len(item_rows)} rows to {OUT_BY_ITEM}")
    print(f"Wrote {len(summary_rows)} rows to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
