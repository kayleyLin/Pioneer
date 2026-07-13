"""Recompute Qwen RQ1 n=50 SBERT results with fixed factual-QA paraphrasing.

This wrapper reuses `32_qwen_analyze_rq1_n50_perturbations_sbert.py` but swaps
the factual-QA perturbed generation file for the repaired version:

    qwen/outputs/rq1_qwen_perturbed_generations_n50_factual_qa_fixed.csv

Outputs are written with a `_fixed_factual` suffix so the original Qwen results
remain untouched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "qwen"
SOURCE = ROOT / "src" / "32_qwen_analyze_rq1_n50_perturbations_sbert.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("qwen_sbert", SOURCE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.PERTURBED_FILES = [
        QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa_fixed.csv",
        QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_math_reasoning.csv",
        QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_code_generation.csv",
        QWEN / "outputs" / "rq1_qwen_perturbed_generations_n50_open_ended_writing.csv",
    ]
    module.BY_ITEM = QWEN / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv"
    module.SUMMARY = QWEN / "outputs" / "sbert_rq1_n50_perturbation_summary_fixed_factual.csv"
    module.HEATMAP = QWEN / "outputs" / "sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual.csv"
    module.UNCORRECTED_SUMMARY = (
        QWEN / "outputs" / "sbert_rq1_n50_uncorrected_perturbation_summary_fixed_factual.csv"
    )
    module.UNCORRECTED_HEATMAP = (
        QWEN / "outputs" / "sbert_rq1_n50_uncorrected_heatmap_drift_fixed_factual.csv"
    )
    module.main()


if __name__ == "__main__":
    main()
