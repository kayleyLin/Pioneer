"""Explore drivers of fixed factual-QA paraphrase drift.

This diagnostic script computes lightweight answer containment/F1 and output
length changes for the repaired factual-QA paraphrase condition. It is intended
as an exploratory bridge between Step 5 and the formal Step 6/7 scripts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

BRANCHES = {
    "outputs": {
        "metrics": ROOT / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "original": ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
        "fixed": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "out": ROOT / "outputs" / "factual_paraphrase_driver_diagnostics_fixed_factual.csv",
    },
    "qwen": {
        "metrics": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "original": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
        "fixed": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "out": ROOT / "qwen" / "outputs" / "factual_paraphrase_driver_diagnostics_fixed_factual.csv",
    },
    "llama": {
        "metrics": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "original": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_factual_qa.csv",
        "fixed": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "out": ROOT / "llama" / "outputs" / "factual_paraphrase_driver_diagnostics_fixed_factual.csv",
    },
}


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: object) -> list[str]:
    return normalize(text).split()


def token_f1(reference: object, output: object) -> float:
    ref = word_tokens(reference)
    out = word_tokens(output)
    if not ref or not out:
        return 0.0
    ref_counts = {}
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in out:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(out)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def containment(reference: object, output: object) -> bool:
    ref = normalize(reference)
    out = normalize(output)
    return bool(ref and ref in out)


def output_length(text: object) -> int:
    return len(word_tokens(text))


def summarize_outputs(df: pd.DataFrame, reference_by_item: dict[str, str], prefix: str) -> pd.DataFrame:
    rows = []
    for item_id, group in df.groupby("item_id"):
        reference = reference_by_item[item_id]
        f1s = [token_f1(reference, output) for output in group["output_text"]]
        contains = [containment(reference, output) for output in group["output_text"]]
        lengths = [output_length(output) for output in group["output_text"]]
        rows.append(
            {
                "item_id": item_id,
                f"{prefix}_mean_token_f1": sum(f1s) / len(f1s),
                f"{prefix}_containment_rate": sum(contains) / len(contains),
                f"{prefix}_mean_output_length_tokens": sum(lengths) / len(lengths),
            }
        )
    return pd.DataFrame(rows)


def run_branch(branch: str) -> pd.DataFrame:
    config = BRANCHES[branch]
    metrics = pd.read_csv(config["metrics"])
    original = pd.read_csv(config["original"])
    fixed = pd.read_csv(config["fixed"])
    fixed = fixed[(fixed["task_type"] == "factual_qa") & (fixed["perturbation_type"] == "paraphrasing")].copy()

    reference_by_item = metrics.set_index("item_id")["reference_answer"].astype(str).to_dict()
    original_summary = summarize_outputs(original, reference_by_item, "original")
    fixed_summary = summarize_outputs(fixed, reference_by_item, "paraphrase")
    diagnostics = metrics.merge(original_summary, on="item_id").merge(fixed_summary, on="item_id")
    diagnostics["token_f1_delta"] = (
        diagnostics["paraphrase_mean_token_f1"] - diagnostics["original_mean_token_f1"]
    )
    diagnostics["containment_rate_delta"] = (
        diagnostics["paraphrase_containment_rate"] - diagnostics["original_containment_rate"]
    )
    diagnostics["output_length_delta_tokens"] = (
        diagnostics["paraphrase_mean_output_length_tokens"]
        - diagnostics["original_mean_output_length_tokens"]
    )

    config["out"].parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(config["out"], index=False)
    print(f"Wrote {config['out'].relative_to(ROOT)}")

    corr_cols = [
        "token_f1_delta",
        "containment_rate_delta",
        "output_length_delta_tokens",
        "cue_disruption",
        "question_content_recall",
        "capitalized_phrase_recall",
        "wh_word_preserved",
    ]
    print(f"\n{branch}")
    print(
        diagnostics[
            [
                "noise_corrected_drift",
                "token_f1_delta",
                "containment_rate_delta",
                "output_length_delta_tokens",
            ]
        ]
        .mean(numeric_only=True)
        .round(6)
        .to_string()
    )
    for col in corr_cols:
        sub = diagnostics[[col, "noise_corrected_drift"]].dropna()
        if len(sub) > 2 and sub[col].nunique() > 1:
            rho, p_value = stats.spearmanr(sub[col], sub["noise_corrected_drift"])
            print(f"{col}: rho={rho:.4f}, p={p_value:.4f}, n={len(sub)}")
    return diagnostics


def main() -> None:
    for branch in BRANCHES:
        run_branch(branch)


if __name__ == "__main__":
    main()
