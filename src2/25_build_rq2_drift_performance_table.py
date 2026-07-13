"""Build RQ2 drift-performance table from available formal outputs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from statistics import mean

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ORIGINAL_GENERATIONS = ROOT / "outputs" / "rq1_formal_original_generations.csv"
PERTURBED_GENERATIONS = ROOT / "outputs" / "rq1_formal_perturbed_generations.csv"
PERFORMANCE_CHANGE = ROOT / "rq2" / "outputs" / "rq2_formal_available_performance_change_by_item.csv"
OUT = ROOT / "rq2" / "outputs" / "rq2_formal_available_drift_performance_by_item.csv"
SUMMARY = ROOT / "rq2" / "outputs" / "rq2_formal_available_drift_performance_summary.csv"
RQ2_TASKS = {"factual_qa", "math_reasoning", "code_generation"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> str:
    if len(xs) < 2 or len(ys) < 2:
        return ""
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return ""
    return f"{num / (den_x * den_y):.6f}"


class SimilarityModel:
    def __init__(self) -> None:
        print(f"Loading Sentence-BERT model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.cache = {}

    def encode(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self.cache[text]

    def similarity(self, a: str, b: str) -> float:
        return float(cos_sim(self.encode(a), self.encode(b))[0][0])

    def within_similarity(self, outputs: list[str]) -> float:
        return safe_mean([self.similarity(a, b) for a, b in combinations(outputs, 2)])

    def cross_similarity(self, originals: list[str], perturbed: list[str]) -> float:
        return safe_mean([self.similarity(a, b) for a, b in product(originals, perturbed)])


def main() -> None:
    original_rows = [row for row in read_csv(ORIGINAL_GENERATIONS) if row["task_type"] in RQ2_TASKS]
    perturbed_rows = [row for row in read_csv(PERTURBED_GENERATIONS) if row["task_type"] in RQ2_TASKS]
    performance_rows = read_csv(PERFORMANCE_CHANGE)

    original_by_item: dict[str, list[str]] = defaultdict(list)
    task_by_item: dict[str, str] = {}
    for row in original_rows:
        original_by_item[row["item_id"]].append(row["output_text"])
        task_by_item[row["item_id"]] = row["task_type"]

    perturbed_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(row["output_text"])

    perf_by_group = {
        (row["item_id"], row["perturbation_type"]): row
        for row in performance_rows
    }

    sim = SimilarityModel()
    baseline_by_item = {
        item_id: sim.within_similarity(outputs)
        for item_id, outputs in original_by_item.items()
    }

    out_rows: list[dict] = []
    for key, perturbed_outputs in sorted(perturbed_by_group.items()):
        item_id, perturbation_type = key
        if item_id not in original_by_item or key not in perf_by_group:
            continue
        baseline = baseline_by_item[item_id]
        perturbation_similarity = sim.cross_similarity(original_by_item[item_id], perturbed_outputs)
        drift = baseline - perturbation_similarity
        perf = perf_by_group[key]
        out_rows.append(
            {
                **perf,
                "baseline_similarity": f"{baseline:.6f}",
                "perturbation_similarity": f"{perturbation_similarity:.6f}",
                "noise_corrected_drift": f"{drift:.6f}",
                "similarity_metric": MODEL_NAME,
            }
        )

    write_csv(
        OUT,
        out_rows,
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
            "baseline_similarity",
            "perturbation_similarity",
            "noise_corrected_drift",
            "similarity_metric",
        ],
    )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in out_rows:
        grouped[(row["task_type"], row["perturbation_type"])].append(row)

    summary_rows: list[dict] = []
    for (task_type, perturbation_type), rows in sorted(grouped.items()):
        drifts = [float(row["noise_corrected_drift"]) for row in rows]
        changes = [float(row["absolute_performance_change"]) for row in rows]
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": str(len(rows)),
                "mean_noise_corrected_drift": f"{mean(drifts):.6f}",
                "mean_absolute_performance_change": f"{mean(changes):.6f}",
                "pearson_drift_performance_change": pearson(drifts, changes),
            }
        )

    write_csv(
        SUMMARY,
        summary_rows,
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_noise_corrected_drift",
            "mean_absolute_performance_change",
            "pearson_drift_performance_change",
        ],
    )
    print(f"Wrote {len(out_rows)} rows to {OUT}")
    print(f"Wrote {len(summary_rows)} rows to {SUMMARY}")


if __name__ == "__main__":
    main()
