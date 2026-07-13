"""Merge repaired math paraphrase generations into full math perturbed files.

This keeps the original files unchanged and writes *_math_reasoning_fixed.csv
files where only the paraphrasing rows are replaced by repaired generations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BRANCHES = {
    "outputs": {
        "original_full": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_math_reasoning.csv",
        "fixed_paraphrase": ROOT
        / "outputs"
        / "rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
        "output": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv",
    },
    "qwen": {
        "original_full": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_math_reasoning.csv",
        "fixed_paraphrase": ROOT
        / "qwen"
        / "outputs"
        / "rq1_qwen_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
        "output": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_math_reasoning_fixed.csv",
    },
    "llama": {
        "original_full": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_math_reasoning.csv",
        "fixed_paraphrase": ROOT
        / "llama"
        / "outputs"
        / "rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
        "output": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_math_reasoning_fixed.csv",
    },
}


def validate(branch: str, df: pd.DataFrame) -> None:
    failures = []
    if len(df) != 1250:
        failures.append(f"expected 1250 rows, found {len(df)}")
    if df["item_id"].nunique() != 50:
        failures.append(f"expected 50 unique items, found {df['item_id'].nunique()}")
    counts = df.groupby(["item_id", "perturbation_type"]).size()
    if len(counts) != 250:
        failures.append(f"expected 250 item-perturbation cells, found {len(counts)}")
    bad_counts = counts[counts != 5]
    if not bad_counts.empty:
        failures.append(f"non-5 sample cells: {bad_counts.head().to_dict()}")
    if df["output_text"].fillna("").astype(str).str.strip().eq("").any():
        failures.append("empty output_text rows")
    if failures:
        raise SystemExit(f"{branch}: validation failed: " + "; ".join(failures))


def main() -> None:
    for branch, config in BRANCHES.items():
        original = pd.read_csv(config["original_full"])
        fixed_para = pd.read_csv(config["fixed_paraphrase"])
        non_para = original[original["perturbation_type"] != "paraphrasing"].copy()
        merged = pd.concat([fixed_para, non_para], ignore_index=True)
        merged = merged.sort_values(["item_id", "perturbation_type", "sample_id"]).reset_index(drop=True)
        validate(branch, merged)
        config["output"].parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(config["output"], index=False)
        print(f"{branch}: wrote {config['output']} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
