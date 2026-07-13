"""Recompute RQ1 n=50 SBERT metrics with fixed factual and math data.

Writes new *_fixed_factual_math.csv files and leaves previous result files
unchanged.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations, product
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BRANCHES = {
    "outputs": {
        "original": [
            ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
            ROOT / "outputs" / "rq1_formal_original_generations_n50_math_reasoning.csv",
            ROOT / "outputs" / "rq1_formal_original_generations_n50_code_generation.csv",
            ROOT / "outputs" / "rq1_formal_original_generations_n50_open_ended_writing.csv",
        ],
        "perturbed": [
            ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa_fixed.csv",
            ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv",
            ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_code_generation.csv",
            ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_open_ended_writing.csv",
        ],
        "outdir": ROOT / "outputs",
    },
    "qwen": {
        "original": [
            ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
            ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_math_reasoning.csv",
            ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_code_generation.csv",
            ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_open_ended_writing.csv",
        ],
        "perturbed": [
            ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa_fixed.csv",
            ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_math_reasoning_fixed.csv",
            ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_code_generation.csv",
            ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_open_ended_writing.csv",
        ],
        "outdir": ROOT / "qwen" / "outputs",
    },
    "llama": {
        "original": [
            ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_factual_qa.csv",
            ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_math_reasoning.csv",
            ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_code_generation.csv",
            ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_open_ended_writing.csv",
        ],
        "perturbed": [
            ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa_fixed_factual.csv",
            ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_math_reasoning_fixed.csv",
            ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_code_generation.csv",
            ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_open_ended_writing.csv",
        ],
        "outdir": ROOT / "llama" / "outputs",
    },
}

PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


class SimilarityModel:
    def __init__(self) -> None:
        print(f"Loading Sentence-BERT model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.cache = {}

    def encode(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self.cache[text]

    def similarity(self, left: str, right: str) -> float:
        return float(cos_sim(self.encode(left), self.encode(right))[0][0])

    def within_similarity(self, outputs: list[str]) -> float:
        return mean([self.similarity(a, b) for a, b in combinations(outputs, 2)])

    def cross_similarity(self, original: list[str], perturbed: list[str]) -> float:
        return mean([self.similarity(a, b) for a, b in product(original, perturbed)])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def load_all(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def validate(original: pd.DataFrame, perturbed: pd.DataFrame) -> None:
    original_counts = original.groupby("item_id").size()
    if len(original_counts) != 200 or not (original_counts == 5).all():
        raise SystemExit("Original rows do not contain 200 items x 5 samples.")
    perturbed_counts = perturbed.groupby(["item_id", "perturbation_type"]).size()
    if len(perturbed_counts) != 1000 or not (perturbed_counts == 5).all():
        raise SystemExit("Perturbed rows do not contain 1000 item-perturbation cells x 5 samples.")
    if original["output_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("Original rows contain empty outputs.")
    if perturbed["output_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("Perturbed rows contain empty outputs.")


def heatmap(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = summary.pivot(index="perturbation_type", columns="task_type", values=value_col)
    return pivot.reindex(PERTURBATION_ORDER).reset_index()


def recompute(branch: str, sim: SimilarityModel) -> None:
    config = BRANCHES[branch]
    original = load_all(config["original"])
    perturbed = load_all(config["perturbed"])
    validate(original, perturbed)

    original_by_item = {
        item_id: group["output_text"].astype(str).tolist()
        for item_id, group in original.groupby("item_id")
    }
    task_by_item = original.groupby("item_id")["task_type"].first().to_dict()
    baseline_by_item = {
        item_id: sim.within_similarity(outputs)
        for item_id, outputs in original_by_item.items()
    }

    item_rows = []
    for (item_id, perturbation_type), group in sorted(perturbed.groupby(["item_id", "perturbation_type"])):
        original_outputs = original_by_item[item_id]
        perturbed_outputs = group["output_text"].astype(str).tolist()
        baseline_similarity = baseline_by_item[item_id]
        perturbation_similarity = sim.cross_similarity(original_outputs, perturbed_outputs)
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": task_by_item[item_id],
                "perturbation_type": perturbation_type,
                "n_original_outputs": len(original_outputs),
                "n_perturbed_outputs": len(perturbed_outputs),
                "baseline_similarity": round(baseline_similarity, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "uncorrected_drift": round(1 - perturbation_similarity, 6),
                "noise_corrected_drift": round(baseline_similarity - perturbation_similarity, 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    by_item = pd.DataFrame(item_rows)
    summary_rows = []
    for (task_type, perturbation_type), group in by_item.groupby(["task_type", "perturbation_type"]):
        values = group["noise_corrected_drift"].astype(float).tolist()
        uncorrected = group["uncorrected_drift"].astype(float).tolist()
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(group),
                "mean_noise_corrected_drift": round(mean(values), 6),
                "std_noise_corrected_drift": round(sample_std(values), 6),
                "mean_uncorrected_drift": round(mean(uncorrected), 6),
                "std_uncorrected_drift": round(sample_std(uncorrected), 6),
                "similarity_metric": MODEL_NAME,
            }
        )
    summary = pd.DataFrame(summary_rows)
    uncorrected_summary = summary[
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_uncorrected_drift",
            "std_uncorrected_drift",
            "similarity_metric",
        ]
    ].copy()

    outdir = config["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    by_item.to_csv(outdir / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual_math.csv", index=False)
    summary[
        [
            "task_type",
            "perturbation_type",
            "n_items",
            "mean_noise_corrected_drift",
            "std_noise_corrected_drift",
            "similarity_metric",
        ]
    ].to_csv(outdir / "sbert_rq1_n50_perturbation_summary_fixed_factual_math.csv", index=False)
    heatmap(summary, "mean_noise_corrected_drift").to_csv(
        outdir / "sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual_math.csv", index=False
    )
    uncorrected_summary.to_csv(
        outdir / "sbert_rq1_n50_uncorrected_perturbation_summary_fixed_factual_math.csv", index=False
    )
    heatmap(summary, "mean_uncorrected_drift").to_csv(
        outdir / "sbert_rq1_n50_uncorrected_heatmap_drift_fixed_factual_math.csv", index=False
    )
    print(f"{branch}: wrote fixed factual+math SBERT outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["outputs", "qwen", "llama", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    sim = SimilarityModel()
    for branch in branches:
        recompute(branch, sim)


if __name__ == "__main__":
    main()
