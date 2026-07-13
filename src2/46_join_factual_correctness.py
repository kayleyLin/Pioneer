"""Compute lightweight factual correctness checks for fixed factual-QA paraphrases.

This is the formal Step 6 script for the repaired factual QA follow-up. It uses
dependency-free reference containment and token-F1 checks at generation level,
then aggregates to one row per item.

Outputs:
    outputs/factual_paraphrase_correctness_by_item_fixed_factual.csv
    qwen/outputs/factual_paraphrase_correctness_by_item_fixed_factual.csv
    llama/outputs/factual_paraphrase_correctness_by_item_fixed_factual.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

BRANCHES = {
    "outputs": {
        "metrics": ROOT / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "original": ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
        "paraphrase": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "output": ROOT / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "summary": ROOT / "outputs" / "factual_paraphrase_correctness_summary_fixed_factual.md",
    },
    "qwen": {
        "metrics": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "original": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
        "paraphrase": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "output": ROOT / "qwen" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "summary": ROOT / "qwen" / "outputs" / "factual_paraphrase_correctness_summary_fixed_factual.md",
    },
    "llama": {
        "metrics": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "original": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_factual_qa.csv",
        "paraphrase": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "output": ROOT / "llama" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "summary": ROOT / "llama" / "outputs" / "factual_paraphrase_correctness_summary_fixed_factual.md",
    },
}


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: object) -> list[str]:
    return normalize(text).split()


def token_f1(reference: object, output: object) -> float:
    ref_tokens = tokens(reference)
    out_tokens = tokens(output)
    if not ref_tokens or not out_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in ref_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in out_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(out_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def containment(reference: object, output: object) -> bool:
    ref = normalize(reference)
    out = normalize(output)
    return bool(ref and ref in out)


def output_length(text: object) -> int:
    return len(tokens(text))


def summarize_generations(
    df: pd.DataFrame,
    reference_by_item: dict[str, str],
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for item_id, group in df.groupby("item_id"):
        reference = reference_by_item[item_id]
        f1_values = [token_f1(reference, output) for output in group["output_text"]]
        containment_values = [containment(reference, output) for output in group["output_text"]]
        length_values = [output_length(output) for output in group["output_text"]]
        rows.append(
            {
                "item_id": item_id,
                f"{prefix}_mean_factual_score": sum(f1_values) / len(f1_values),
                f"{prefix}_containment_rate": sum(containment_values) / len(containment_values),
                f"{prefix}_mean_output_length_tokens": sum(length_values) / len(length_values),
            }
        )
    return pd.DataFrame(rows)


def spearman(df: pd.DataFrame, x_col: str, y_col: str = "noise_corrected_drift") -> tuple[float, float, int]:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 3 or sub[x_col].nunique() < 2:
        return float("nan"), float("nan"), len(sub)
    rho, p_value = stats.spearmanr(sub[x_col], sub[y_col])
    return float(rho), float(p_value), len(sub)


def validate(branch: str, df: pd.DataFrame) -> None:
    failures = []
    if len(df) != 50:
        failures.append(f"expected 50 rows, found {len(df)}")
    if df["item_id"].nunique() != 50:
        failures.append(f"expected 50 unique items, found {df['item_id'].nunique()}")
    required = [
        "original_mean_factual_score",
        "paraphrase_mean_factual_score",
        "factual_score_delta",
        "original_containment_rate",
        "paraphrase_containment_rate",
        "containment_rate_delta",
    ]
    for col in required:
        if col not in df.columns:
            failures.append(f"missing column {col}")
        elif df[col].isna().any():
            failures.append(f"{col} contains NA")
    if not df["original_containment_rate"].between(0, 1).all():
        failures.append("original_containment_rate outside [0,1]")
    if not df["paraphrase_containment_rate"].between(0, 1).all():
        failures.append("paraphrase_containment_rate outside [0,1]")
    if failures:
        raise SystemExit(f"{branch}: validation failed: " + "; ".join(failures))


def write_summary(branch: str, df: pd.DataFrame, path: Path) -> None:
    corr_rows = []
    for col in [
        "factual_score_delta",
        "containment_rate_delta",
        "output_length_delta_tokens",
        "cue_disruption",
        "question_content_recall",
        "question_context_content_overlap_delta",
    ]:
        rho, p_value, n = spearman(df, col)
        corr_rows.append({"metric": col, "spearman_rho": rho, "p_value": p_value, "n": n})
    corr = pd.DataFrame(corr_rows)
    high = df.nlargest(5, "noise_corrected_drift")[
        [
            "item_id",
            "reference_answer",
            "noise_corrected_drift",
            "factual_score_delta",
            "containment_rate_delta",
            "output_length_delta_tokens",
            "cue_disruption",
        ]
    ]
    lines = [
        f"# Factual Paraphrase Correctness Summary: {branch}",
        "",
        f"- Rows: {len(df)}",
        f"- Mean original token-F1 score: {df['original_mean_factual_score'].mean():.6f}",
        f"- Mean paraphrase token-F1 score: {df['paraphrase_mean_factual_score'].mean():.6f}",
        f"- Mean factual score delta: {df['factual_score_delta'].mean():.6f}",
        f"- Mean original containment rate: {df['original_containment_rate'].mean():.6f}",
        f"- Mean paraphrase containment rate: {df['paraphrase_containment_rate'].mean():.6f}",
        f"- Mean containment rate delta: {df['containment_rate_delta'].mean():.6f}",
        f"- Mean output length delta tokens: {df['output_length_delta_tokens'].mean():.6f}",
        "",
        "## Correlations With Noise-Corrected Drift",
        "",
        corr.to_markdown(index=False),
        "",
        "## Highest Drift Items",
        "",
        high.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_branch(branch: str) -> pd.DataFrame:
    config = BRANCHES[branch]
    metrics = pd.read_csv(config["metrics"])
    original = pd.read_csv(config["original"])
    paraphrase = pd.read_csv(config["paraphrase"])
    paraphrase = paraphrase[
        (paraphrase["task_type"] == "factual_qa")
        & (paraphrase["perturbation_type"] == "paraphrasing")
    ].copy()

    reference_by_item = metrics.set_index("item_id")["reference_answer"].astype(str).to_dict()
    original_summary = summarize_generations(original, reference_by_item, "original")
    paraphrase_summary = summarize_generations(paraphrase, reference_by_item, "paraphrase")
    result = metrics.merge(original_summary, on="item_id").merge(paraphrase_summary, on="item_id")
    result["factual_score_delta"] = (
        result["paraphrase_mean_factual_score"] - result["original_mean_factual_score"]
    )
    result["containment_rate_delta"] = (
        result["paraphrase_containment_rate"] - result["original_containment_rate"]
    )
    result["output_length_delta_tokens"] = (
        result["paraphrase_mean_output_length_tokens"]
        - result["original_mean_output_length_tokens"]
    )
    validate(branch, result)
    config["output"].parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config["output"], index=False)
    write_summary(branch, result, config["summary"])
    print(f"Wrote {config['output'].relative_to(ROOT)}")
    print(f"Wrote {config['summary'].relative_to(ROOT)}")
    print(
        f"{branch}: mean factual_score_delta={result['factual_score_delta'].mean():.6f}, "
        f"mean containment_rate_delta={result['containment_rate_delta'].mean():.6f}, "
        f"mean output_length_delta_tokens={result['output_length_delta_tokens'].mean():.6f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["outputs", "qwen", "llama", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    for branch in branches:
        run_branch(branch)


if __name__ == "__main__":
    main()
