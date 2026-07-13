"""Recompute the Llama factual-QA paraphrasing SBERT cell after repair."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "llama" / "outputs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ORIGINAL = OUT / "rq1_llama_original_generations_n50_factual_qa.csv"
FIXED_PARAPHRASE = OUT / "rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv"
OLD_EFFECTS = OUT / "sbert_rq1_n50_perturbation_effects_by_item.csv"

CELL_EFFECTS = OUT / "sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv"
MERGED_EFFECTS = OUT / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv"
SUMMARY = OUT / "sbert_rq1_n50_perturbation_summary_fixed_factual.csv"
HEATMAP = OUT / "sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual.csv"
UNCORRECTED_SUMMARY = OUT / "sbert_rq1_n50_uncorrected_perturbation_summary_fixed_factual.csv"
UNCORRECTED_HEATMAP = OUT / "sbert_rq1_n50_uncorrected_heatmap_drift_fixed_factual.csv"

TASK = "factual_qa"
PERTURBATION = "paraphrasing"
PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def validate_generation_rows(original: pd.DataFrame, fixed: pd.DataFrame) -> None:
    if len(original) != 250 or len(fixed) != 250:
        raise SystemExit(f"Expected 250 original and 250 fixed rows, found {len(original)} and {len(fixed)}")
    if original["item_id"].nunique() != 50 or fixed["item_id"].nunique() != 50:
        raise SystemExit("Original or fixed rows do not cover 50 factual-QA items.")
    if original["output_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("Original rows contain empty outputs.")
    if fixed["output_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("Fixed paraphrase rows contain empty outputs.")
    if not original.groupby("item_id")["sample_id"].nunique().eq(5).all():
        raise SystemExit("Some original items do not have 5 samples.")
    if not fixed.groupby("item_id")["sample_id"].nunique().eq(5).all():
        raise SystemExit("Some fixed paraphrase items do not have 5 samples.")
    if set(original["item_id"]) != set(fixed["item_id"]):
        raise SystemExit("Original and fixed paraphrase item sets differ.")


def encode_outputs(model: SentenceTransformer, original: pd.DataFrame, fixed: pd.DataFrame) -> dict[str, object]:
    all_texts = pd.concat([original["output_text"], fixed["output_text"]], ignore_index=True).astype(str).tolist()
    unique_texts = list(dict.fromkeys(all_texts))
    embeddings = model.encode(unique_texts, convert_to_tensor=True, show_progress_bar=False)
    return dict(zip(unique_texts, embeddings))


def within_similarity(outputs: list[str], embeddings: dict[str, object]) -> float:
    return mean(
        [
            float(cos_sim(embeddings[output_a], embeddings[output_b])[0][0])
            for output_a, output_b in combinations(outputs, 2)
        ]
    )


def cross_similarity(original_outputs: list[str], fixed_outputs: list[str], embeddings: dict[str, object]) -> float:
    return mean(
        [
            float(cos_sim(embeddings[original_output], embeddings[fixed_output])[0][0])
            for original_output in original_outputs
            for fixed_output in fixed_outputs
        ]
    )


def recompute_cell() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL)
    fixed = pd.read_csv(FIXED_PARAPHRASE)
    original = original[original["task_type"] == TASK].copy()
    fixed = fixed[(fixed["task_type"] == TASK) & (fixed["perturbation_type"] == PERTURBATION)].copy()
    validate_generation_rows(original, fixed)

    print(f"Loading Sentence-BERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = encode_outputs(model, original, fixed)

    rows = []
    for item_id in sorted(original["item_id"].unique()):
        original_outputs = original[original["item_id"] == item_id].sort_values("sample_id")["output_text"].astype(str).tolist()
        fixed_outputs = fixed[fixed["item_id"] == item_id].sort_values("sample_id")["output_text"].astype(str).tolist()
        baseline_similarity = within_similarity(original_outputs, embeddings)
        perturbation_similarity = cross_similarity(original_outputs, fixed_outputs, embeddings)
        rows.append(
            {
                "item_id": item_id,
                "task_type": TASK,
                "perturbation_type": PERTURBATION,
                "n_original_outputs": len(original_outputs),
                "n_perturbed_outputs": len(fixed_outputs),
                "baseline_similarity": round(baseline_similarity, 6),
                "perturbation_similarity": round(perturbation_similarity, 6),
                "uncorrected_drift": round(1 - perturbation_similarity, 6),
                "noise_corrected_drift": round(baseline_similarity - perturbation_similarity, 6),
                "similarity_metric": MODEL_NAME,
            }
        )
    cell = pd.DataFrame(rows)
    cell.to_csv(CELL_EFFECTS, index=False)
    return cell


def heatmap_rows(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tasks = sorted(summary["task_type"].unique())
    pivot = summary.pivot(index="perturbation_type", columns="task_type", values=value_col)
    return pivot.reindex(index=PERTURBATION_ORDER, columns=tasks).reset_index()


def rebuild_outputs(cell: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_EFFECTS)
    keep = ~((old["task_type"] == TASK) & (old["perturbation_type"] == PERTURBATION))
    merged = pd.concat([old[keep], cell], ignore_index=True).sort_values(
        ["task_type", "perturbation_type", "item_id"]
    )
    if len(merged) != 1000:
        raise SystemExit(f"Merged effects should have 1000 rows, found {len(merged)}")
    merged.to_csv(MERGED_EFFECTS, index=False)

    summary_rows = []
    uncorrected_rows = []
    for (task_type, perturbation_type), group in merged.groupby(["task_type", "perturbation_type"]):
        corrected = group["noise_corrected_drift"].astype(float).tolist()
        uncorrected = group["uncorrected_drift"].astype(float).tolist()
        summary_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(group),
                "mean_noise_corrected_drift": round(mean(corrected), 6),
                "std_noise_corrected_drift": round(sample_std(corrected), 6),
                "similarity_metric": MODEL_NAME,
            }
        )
        uncorrected_rows.append(
            {
                "task_type": task_type,
                "perturbation_type": perturbation_type,
                "n_items": len(group),
                "mean_uncorrected_drift": round(mean(uncorrected), 6),
                "std_uncorrected_drift": round(sample_std(uncorrected), 6),
                "similarity_metric": MODEL_NAME,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["task_type", "perturbation_type"])
    uncorrected_summary = pd.DataFrame(uncorrected_rows).sort_values(["task_type", "perturbation_type"])
    summary.to_csv(SUMMARY, index=False)
    uncorrected_summary.to_csv(UNCORRECTED_SUMMARY, index=False)
    heatmap_rows(summary, "mean_noise_corrected_drift").to_csv(HEATMAP, index=False)
    heatmap_rows(uncorrected_summary, "mean_uncorrected_drift").to_csv(UNCORRECTED_HEATMAP, index=False)
    return summary


def main() -> None:
    cell = recompute_cell()
    summary = rebuild_outputs(cell)
    target = summary[(summary["task_type"] == TASK) & (summary["perturbation_type"] == PERTURBATION)].iloc[0]
    print(f"Wrote {CELL_EFFECTS.relative_to(ROOT)}")
    print(f"Wrote {MERGED_EFFECTS.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "Fixed Llama factual_qa + paraphrasing mean NCP: "
        f"{target['mean_noise_corrected_drift']:.6f}"
    )


if __name__ == "__main__":
    main()
