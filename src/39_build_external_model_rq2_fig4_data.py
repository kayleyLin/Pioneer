"""Build real RQ2 correctness/performance data for external model Figure 4.

This reuses the project's automatic correctness evaluator:
- factual QA: containment-first token F1 performance score
- math reasoning: final-answer correctness
- code generation: HumanEvalPack functional tests

The script writes outputs inside each model directory, e.g.
data/llama/rq2_outputs/, so it does not overwrite the original GPT RQ2 files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50.csv"
RQ2_TASKS = {"factual_qa", "math_reasoning", "code_generation"}
TASKS = ["factual_qa", "math_reasoning", "code_generation"]


def load_evaluator():
    path = ROOT / "src" / "23_evaluate_rq2_correctness.py"
    spec = importlib.util.spec_from_file_location("rq2_correctness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    if fieldnames is None:
        raise ValueError(f"No fieldnames for empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_model_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def prefix_from_model_dir(model_dir: Path, explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return explicit_prefix
    name = model_dir.name.lower()
    if name == "llama":
        return "rq1_llama"
    if name == "qwen":
        return "rq1_qwen"
    raise SystemExit("--prefix is required for model directories other than llama/qwen")


def generation_files(model_dir: Path, prefix: str, variant: str) -> list[Path]:
    return [
        model_dir / "outputs" / f"{prefix}_{variant}_generations_n50_{task}.csv"
        for task in ["factual_qa", "math_reasoning", "code_generation"]
    ]


def evaluate_generations(
    evaluator,
    generation_rows: list[dict[str, str]],
    prompt_by_item: dict[str, dict[str, str]],
    prompt_variant: str,
) -> list[dict[str, str]]:
    code_source_indices = {
        row.get("source_index", "")
        for row in generation_rows
        if row.get("task_type") == "code_generation"
    }
    code_tests = evaluator.load_humanevalpack_tests(code_source_indices)

    output_rows = []
    for index, row in enumerate(generation_rows, start=1):
        prompt_meta = prompt_by_item.get(row["item_id"], {})
        result = evaluator.evaluate_row(row, prompt_meta, code_tests)
        output_rows.append(
            {
                "item_id": row.get("item_id", ""),
                "task_type": row.get("task_type", ""),
                "dataset_name": row.get("dataset_name", ""),
                "source_index": row.get("source_index", ""),
                "source_id": prompt_meta.get("source_id", ""),
                "sample_id": row.get("sample_id", ""),
                "model_name": row.get("model_name", ""),
                "prompt_variant": prompt_variant,
                "perturbation_type": row.get("perturbation_type", ""),
                "reference_answer": prompt_meta.get("reference_answer", ""),
                "output_text": row.get("output_text", ""),
                "extracted_answer": result.extracted_answer,
                "performance_score": ""
                if result.performance_score is None
                else f"{result.performance_score:.6f}",
                "factual_containment_match": ""
                if result.factual_containment_match is None
                else ("true" if result.factual_containment_match else "false"),
                "factual_token_f1": ""
                if result.factual_token_f1 is None
                else f"{result.factual_token_f1:.6f}",
                "is_correct": evaluator.bool_to_csv(result.is_correct),
                "needs_manual_review": "true" if result.needs_manual_review else "false",
                "correctness_method": result.correctness_method,
                "notes": result.notes,
            }
        )
        if index % 250 == 0:
            print(f"Evaluated {index}/{len(generation_rows)} {prompt_variant} rows")
    return output_rows


def summarize_correctness(rows: list[dict[str, str]], evaluator) -> list[dict[str, str]]:
    return evaluator.summarize(rows)


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


def build_performance_change(
    original_rows: list[dict[str, str]],
    perturbed_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    original_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in original_rows:
        original_by_item[row["item_id"]].append(row)

    perturbed_by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(row)

    item_rows = []
    for (item_id, perturbation_type), rows in sorted(perturbed_by_group.items()):
        if item_id not in original_by_item:
            continue
        original_score = mean_score(original_by_item[item_id])
        perturbed_score = mean_score(rows)
        absolute_change = original_score - perturbed_score
        item_rows.append(
            {
                "item_id": item_id,
                "task_type": rows[0]["task_type"],
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
    return item_rows


def build_drift_performance(
    performance_rows: list[dict[str, str]],
    model_dir: Path,
) -> list[dict[str, str]]:
    effects_path = model_dir / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item.csv"
    effects = pd.read_csv(effects_path)
    effect_by_group = {
        (row.item_id, row.perturbation_type): row
        for row in effects.itertuples(index=False)
        if row.task_type in RQ2_TASKS
    }

    out_rows = []
    for row in performance_rows:
        key = (row["item_id"], row["perturbation_type"])
        if key not in effect_by_group:
            continue
        effect = effect_by_group[key]
        out_rows.append(
            {
                **row,
                "baseline_similarity": f"{float(effect.baseline_similarity):.6f}",
                "perturbation_similarity": f"{float(effect.perturbation_similarity):.6f}",
                "uncorrected_drift": f"{float(effect.uncorrected_drift):.6f}",
                "noise_corrected_drift": f"{float(effect.noise_corrected_drift):.6f}",
                "similarity_metric": str(effect.similarity_metric),
            }
        )
    return out_rows


def build_fig4_data(drift_performance_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fig_rows = []
    for row in drift_performance_rows:
        original_performance = float(row["original_performance"])
        perturbed_performance = float(row["perturbed_performance"])
        if original_performance > 0:
            correctness_retention = min(perturbed_performance / original_performance, 1.0)
            correctness_retention_text = f"{correctness_retention:.6f}"
        else:
            correctness_retention_text = ""
        fig_rows.append(
            {
                "item_id": row["item_id"],
                "task_type": row["task_type"],
                "perturbation_type": row["perturbation_type"],
                "similarity": row["perturbation_similarity"],
                "perturbed_performance": row["perturbed_performance"],
                "original_performance": row["original_performance"],
                "correctness_retention": correctness_retention_text,
                "absolute_performance_change": row["absolute_performance_change"],
                "pdr": row["pdr"],
                "similarity_metric": row["similarity_metric"],
            }
        )

    df = pd.DataFrame(fig_rows)
    df["similarity"] = df["similarity"].astype(float)
    df["correctness_retention"] = pd.to_numeric(df["correctness_retention"], errors="coerce").clip(0, 1)
    bins = [-float("inf"), 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.000001]
    labels = ["<0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
    df["similarity_bin"] = pd.cut(df["similarity"], bins=bins, labels=labels, include_lowest=True, right=False)
    valid = df.dropna(subset=["correctness_retention"]).copy()
    summary = (
        valid.groupby("similarity_bin", observed=False)
        .agg(
            mean_correctness_retention=("correctness_retention", "mean"),
            n=("correctness_retention", "size"),
        )
        .reset_index()
    )
    summary = summary[summary["n"] > 0].copy()
    summary_rows = [
        {
            "similarity_bin": str(row.similarity_bin),
            "mean_correctness_retention": f"{float(row.mean_correctness_retention):.6f}",
            "n": str(int(row.n)),
        }
        for row in summary.itertuples(index=False)
    ]
    return fig_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    prefix = prefix_from_model_dir(model_dir, args.prefix)
    out_dir = model_dir / "rq2_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = load_evaluator()
    prompt_rows = read_csv(PROMPTS)
    prompt_by_item = {row["item_id"]: row for row in prompt_rows}

    original_generation_rows = []
    for path in generation_files(model_dir, prefix, "original"):
        original_generation_rows.extend(read_csv(path))
    perturbed_generation_rows = []
    for path in generation_files(model_dir, prefix, "perturbed"):
        perturbed_generation_rows.extend(read_csv(path))

    original_generation_rows = [
        row for row in original_generation_rows if row.get("task_type") in RQ2_TASKS
    ]
    perturbed_generation_rows = [
        row for row in perturbed_generation_rows if row.get("task_type") in RQ2_TASKS
    ]

    original_correctness = evaluate_generations(
        evaluator, original_generation_rows, prompt_by_item, "original"
    )
    perturbed_correctness = evaluate_generations(
        evaluator, perturbed_generation_rows, prompt_by_item, "perturbed"
    )
    correctness_fieldnames = [
        "item_id",
        "task_type",
        "dataset_name",
        "source_index",
        "source_id",
        "sample_id",
        "model_name",
        "prompt_variant",
        "perturbation_type",
        "reference_answer",
        "output_text",
        "extracted_answer",
        "performance_score",
        "factual_containment_match",
        "factual_token_f1",
        "is_correct",
        "needs_manual_review",
        "correctness_method",
        "notes",
    ]
    write_csv(out_dir / "rq2_original_correctness_by_generation.csv", original_correctness, correctness_fieldnames)
    write_csv(out_dir / "rq2_perturbed_correctness_by_generation.csv", perturbed_correctness, correctness_fieldnames)
    write_csv(out_dir / "rq2_original_correctness_summary_by_task.csv", summarize_correctness(original_correctness, evaluator))
    write_csv(out_dir / "rq2_perturbed_correctness_summary_by_task.csv", summarize_correctness(perturbed_correctness, evaluator))

    performance_change = build_performance_change(original_correctness, perturbed_correctness)
    write_csv(out_dir / "rq2_performance_change_by_item.csv", performance_change)
    drift_performance = build_drift_performance(performance_change, model_dir)
    write_csv(out_dir / "rq2_drift_performance_by_item.csv", drift_performance)
    fig4_rows, fig4_summary = build_fig4_data(drift_performance)
    write_csv(out_dir / "fig4_similarity_correctness_data.csv", fig4_rows)
    write_csv(out_dir / "fig4_similarity_correctness_binned.csv", fig4_summary)

    print(f"Wrote RQ2 external outputs to {out_dir}")
    print(f"Original correctness rows: {len(original_correctness)}")
    print(f"Perturbed correctness rows: {len(perturbed_correctness)}")
    print(f"Figure 4 rows: {len(fig4_rows)}")


if __name__ == "__main__":
    main()
