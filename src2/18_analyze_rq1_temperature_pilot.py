"""Analyze RQ1 temperature-pilot outputs with Sentence-BERT similarity."""

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GENERATIONS = ROOT / "outputs" / "rq1_temperature_pilot_generations.csv"
BY_ITEM = ROOT / "outputs" / "sbert_rq1_temperature_pilot_by_item.csv"
BY_TASK = ROOT / "outputs" / "sbert_rq1_temperature_pilot_by_task.csv"
OVERALL = ROOT / "outputs" / "sbert_rq1_temperature_pilot_overall.csv"
COARSE = ROOT / "outputs" / "sbert_rq1_temperature_pilot_coarse_summary.csv"
LOCAL = ROOT / "outputs" / "sbert_rq1_temperature_pilot_local_summary.csv"

COARSE_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
LOCAL_TEMPERATURES = [0.5, 0.7, 0.8, 0.9]


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


def temperature_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_task_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    overall_values: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        task_type = row["task_type"]
        temperature = row["temperature"]
        drift = float(row["sampling_noise_drift"])
        by_task_values[(task_type, temperature)].append(drift)
        overall_values[temperature].append(drift)

    by_task_rows = []
    for (task_type, temperature), values in sorted(
        by_task_values.items(), key=lambda item: (item[0][0], float(item[0][1]))
    ):
        by_task_rows.append(
            {
                "task_type": task_type,
                "temperature": temperature,
                "n_items": len(values),
                "mean_sampling_noise_drift": round(mean(values), 6),
                "std_sampling_noise_drift": round(sample_std(values), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    overall_rows = []
    for temperature, values in sorted(
        overall_values.items(), key=lambda item: float(item[0])
    ):
        overall_rows.append(
            {
                "temperature": temperature,
                "n_items": len(values),
                "mean_sampling_noise_drift": round(mean(values), 6),
                "std_sampling_noise_drift": round(sample_std(values), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    return rows, by_task_rows, overall_rows


def phase_summary(overall_rows: list[dict], temperatures: list[float]) -> list[dict]:
    lookup = {float(row["temperature"]): row for row in overall_rows}
    rows = []
    previous_drift = None
    for temperature in temperatures:
        row = lookup.get(temperature)
        if not row:
            continue
        drift = float(row["mean_sampling_noise_drift"])
        rows.append(
            {
                "temperature": temperature_label(temperature),
                "n_items": row["n_items"],
                "mean_sampling_noise_drift": round(drift, 6),
                "change_from_previous_temperature": ""
                if previous_drift is None
                else round(drift - previous_drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )
        previous_drift = drift
    return rows


def main() -> None:
    rows = read_csv(GENERATIONS)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = (row["item_id"], row["task_type"], temperature_label(float(row["temperature"])))
        grouped[key].append(row)

    sim_model = SimilarityModel()
    item_rows = []

    for (item_id, task_type, temperature), group_rows in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0], float(item[0][2]))
    ):
        group_rows = sorted(group_rows, key=lambda row: int(row["sample_id"]))
        outputs = [row["output_text"] for row in group_rows]
        if len(outputs) < 2:
            continue
        avg_similarity = sim_model.within_similarity(outputs)
        drift = 1 - avg_similarity
        phases = sorted({row.get("temperature_phase", "") for row in group_rows})
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_type,
                "temperature": temperature,
                "temperature_phase": ";".join(phases),
                "n_outputs": len(outputs),
                "n_pairs": len(outputs) * (len(outputs) - 1) // 2,
                "mean_within_prompt_similarity": round(avg_similarity, 6),
                "sampling_noise_drift": round(drift, 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    _, by_task_rows, overall_rows = summarize(item_rows)

    write_csv(
        BY_ITEM,
        item_rows,
        [
            "item_id",
            "task_type",
            "temperature",
            "temperature_phase",
            "n_outputs",
            "n_pairs",
            "mean_within_prompt_similarity",
            "sampling_noise_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        BY_TASK,
        by_task_rows,
        [
            "task_type",
            "temperature",
            "n_items",
            "mean_sampling_noise_drift",
            "std_sampling_noise_drift",
            "similarity_metric",
        ],
    )
    write_csv(
        OVERALL,
        overall_rows,
        [
            "temperature",
            "n_items",
            "mean_sampling_noise_drift",
            "std_sampling_noise_drift",
            "similarity_metric",
        ],
    )

    phase_fieldnames = [
        "temperature",
        "n_items",
        "mean_sampling_noise_drift",
        "change_from_previous_temperature",
        "similarity_metric",
    ]
    write_csv(COARSE, phase_summary(overall_rows, COARSE_TEMPERATURES), phase_fieldnames)
    write_csv(LOCAL, phase_summary(overall_rows, LOCAL_TEMPERATURES), phase_fieldnames)

    print(f"Wrote {BY_ITEM}")
    print(f"Wrote {BY_TASK}")
    print(f"Wrote {OVERALL}")
    print(f"Wrote {COARSE}")
    print(f"Wrote {LOCAL}")


if __name__ == "__main__":
    main()
