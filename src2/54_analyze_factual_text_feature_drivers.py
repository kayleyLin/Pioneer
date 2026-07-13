"""Analyze text-level drivers of fixed factual-QA paraphrase drift.

This script is a lightweight bridge between qualitative case analysis and
model-internal probability/attention probes. It uses only existing text,
generation, cue, and correctness files.

Outputs for the selected branch:
    factual_text_feature_base_fixed_factual.csv
    factual_text_feature_driver_correlations_fixed_factual.csv
    factual_text_feature_driver_regressions_fixed_factual.csv
    factual_text_feature_driver_summary_fixed_factual.md
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
from itertools import product
from pathlib import Path

import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except Exception as exc:  # pragma: no cover - fallback path
    sm = None
    STATSMODELS_ERROR = exc
else:
    STATSMODELS_ERROR = None


ROOT = Path(__file__).resolve().parents[1]

BRANCHES = {
    "outputs": {
        "label": "GPT/main",
        "dir": ROOT / "outputs",
        "item": ROOT / "outputs" / "factual_paraphrase_item_table_fixed_factual.csv",
        "cue": ROOT / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "correctness": ROOT / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "original": ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
        "fixed": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
    },
    "qwen": {
        "label": "Qwen",
        "dir": ROOT / "qwen" / "outputs",
        "item": ROOT / "qwen" / "outputs" / "factual_paraphrase_item_table_fixed_factual.csv",
        "cue": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "correctness": ROOT / "qwen" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "original": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
        "fixed": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
    },
    "llama": {
        "label": "Llama",
        "dir": ROOT / "llama" / "outputs",
        "item": ROOT / "llama" / "outputs" / "factual_paraphrase_item_table_fixed_factual.csv",
        "cue": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
        "correctness": ROOT / "llama" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "original": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_factual_qa.csv",
        "fixed": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
    },
}


PROMPT_FEATURES = [
    "prompt_char_edit_distance_norm",
    "question_char_edit_distance_norm",
    "prompt_token_edit_distance_norm",
    "question_token_edit_distance_norm",
    "prompt_length_delta_tokens",
    "question_length_delta_tokens",
    "question_length_ratio",
    "cue_disruption",
    "question_content_recall",
    "question_context_content_overlap_delta",
]

OUTPUT_FEATURES = [
    "output_length_delta_tokens",
    "output_length_ratio",
    "mean_output_token_edit_distance_norm",
    "median_output_token_edit_distance_norm",
    "mean_output_char_edit_distance_norm",
    "factual_score_delta",
    "containment_rate_delta",
    "answer_compactness_loss_proxy",
    "style_expansion_proxy",
    "answer_scope_proxy",
]

REGRESSION_SPECS = [
    (
        "minimal_text_driver",
        [
            "output_length_delta_tokens",
            "factual_score_delta",
            "question_token_edit_distance_norm",
            "cue_disruption",
        ],
    ),
    (
        "scope_style",
        [
            "answer_scope_proxy",
            "question_content_recall",
            "question_context_content_overlap_delta",
        ],
    ),
    (
        "output_distance",
        [
            "mean_output_token_edit_distance_norm",
            "question_token_edit_distance_norm",
            "containment_rate_delta",
        ],
    ),
]


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: object) -> list[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []


def levenshtein(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Return Levenshtein edit distance for token lists or strings."""
    if seq_a == seq_b:
        return 0
    len_a = len(seq_a)
    len_b = len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    for i, a_val in enumerate(seq_a, start=1):
        cur = [i] + [0] * len_b
        for j, b_val in enumerate(seq_b, start=1):
            cost = 0 if a_val == b_val else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len_b]


def norm_distance(distance: int, len_a: int, len_b: int) -> float:
    denom = max(len_a, len_b)
    if denom == 0:
        return 0.0
    return distance / denom


def approximate_char_distance_norm(text_a: str, text_b: str) -> tuple[float, float]:
    """Fast normalized character distance proxy for long generated texts.

    Full character-level Levenshtein is expensive for long answers and is not
    the primary metric here. Token-level Levenshtein is the main edit-distance
    feature; this proxy keeps the char-level column available for inspection.
    """
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 0.0, 0.0
    ratio = difflib.SequenceMatcher(None, text_a, text_b, autojunk=True).ratio()
    distance_norm = 1.0 - ratio
    return distance_norm * max_len, distance_norm


def length_ratio(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return math.nan
    return new_value / old_value


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (numeric - numeric.mean()) / std


def summarize_pairwise_output_distances(original_group: pd.DataFrame, fixed_group: pd.DataFrame) -> dict[str, float]:
    char_distances = []
    char_distances_norm = []
    token_distances = []
    token_distances_norm = []

    for original_output, fixed_output in product(original_group["output_text"], fixed_group["output_text"]):
        original_text = normalize_text(original_output)
        fixed_text = normalize_text(fixed_output)
        original_tokens = original_text.split() if original_text else []
        fixed_tokens = fixed_text.split() if fixed_text else []

        char_distance, char_distance_norm = approximate_char_distance_norm(original_text, fixed_text)
        token_distance = levenshtein(original_tokens, fixed_tokens)

        char_distances.append(char_distance)
        char_distances_norm.append(char_distance_norm)
        token_distances.append(token_distance)
        token_distances_norm.append(norm_distance(token_distance, len(original_tokens), len(fixed_tokens)))

    return {
        "mean_output_char_edit_distance": float(pd.Series(char_distances).mean()),
        "mean_output_char_edit_distance_norm": float(pd.Series(char_distances_norm).mean()),
        "mean_output_token_edit_distance": float(pd.Series(token_distances).mean()),
        "mean_output_token_edit_distance_norm": float(pd.Series(token_distances_norm).mean()),
        "median_output_token_edit_distance_norm": float(pd.Series(token_distances_norm).median()),
        "min_output_token_edit_distance_norm": float(pd.Series(token_distances_norm).min()),
        "max_output_token_edit_distance_norm": float(pd.Series(token_distances_norm).max()),
        "n_output_pairs": len(token_distances_norm),
    }


def add_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        original_prompt = str(row["original_prompt"])
        perturbed_prompt = str(row["perturbed_prompt"])
        original_question = str(row["original_question"])
        paraphrased_question = str(row["paraphrased_question"])

        original_prompt_tokens = tokens(original_prompt)
        perturbed_prompt_tokens = tokens(perturbed_prompt)
        original_question_tokens = tokens(original_question)
        paraphrased_question_tokens = tokens(paraphrased_question)

        norm_original_prompt = normalize_text(original_prompt)
        norm_perturbed_prompt = normalize_text(perturbed_prompt)
        norm_original_question = normalize_text(original_question)
        norm_paraphrased_question = normalize_text(paraphrased_question)
        prompt_char_distance, prompt_char_distance_norm = approximate_char_distance_norm(
            norm_original_prompt,
            norm_perturbed_prompt,
        )
        question_char_distance, question_char_distance_norm = approximate_char_distance_norm(
            norm_original_question,
            norm_paraphrased_question,
        )
        prompt_token_distance = levenshtein(original_prompt_tokens, perturbed_prompt_tokens)
        question_token_distance = levenshtein(original_question_tokens, paraphrased_question_tokens)

        rows.append(
            {
                "item_id": row["item_id"],
                "prompt_char_edit_distance": prompt_char_distance,
                "prompt_char_edit_distance_norm": prompt_char_distance_norm,
                "question_char_edit_distance": question_char_distance,
                "question_char_edit_distance_norm": question_char_distance_norm,
                "prompt_token_edit_distance": prompt_token_distance,
                "prompt_token_edit_distance_norm": norm_distance(
                    prompt_token_distance,
                    len(original_prompt_tokens),
                    len(perturbed_prompt_tokens),
                ),
                "question_token_edit_distance": question_token_distance,
                "question_token_edit_distance_norm": norm_distance(
                    question_token_distance,
                    len(original_question_tokens),
                    len(paraphrased_question_tokens),
                ),
                "original_prompt_length_tokens": len(original_prompt_tokens),
                "perturbed_prompt_length_tokens": len(perturbed_prompt_tokens),
                "prompt_length_delta_tokens": len(perturbed_prompt_tokens) - len(original_prompt_tokens),
                "original_question_length_tokens": len(original_question_tokens),
                "paraphrased_question_length_tokens": len(paraphrased_question_tokens),
                "question_length_delta_tokens": len(paraphrased_question_tokens) - len(original_question_tokens),
                "question_length_ratio": length_ratio(len(paraphrased_question_tokens), len(original_question_tokens)),
            }
        )
    return df.merge(pd.DataFrame(rows), on="item_id", how="left")


def add_output_features(df: pd.DataFrame, original: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    fixed = fixed[(fixed["task_type"] == "factual_qa") & (fixed["perturbation_type"] == "paraphrasing")].copy()
    rows = []
    for item_id, row in df.set_index("item_id").iterrows():
        original_group = original[original["item_id"] == item_id]
        fixed_group = fixed[fixed["item_id"] == item_id]
        distances = summarize_pairwise_output_distances(original_group, fixed_group)
        original_length = float(row["original_mean_output_length_tokens"])
        paraphrase_length = float(row["paraphrase_mean_output_length_tokens"])
        distances.update(
            {
                "item_id": item_id,
                "output_length_ratio": length_ratio(paraphrase_length, original_length),
            }
        )
        rows.append(distances)
    return df.merge(pd.DataFrame(rows), on="item_id", how="left")


def spearman(df: pd.DataFrame, feature: str) -> dict[str, object]:
    sub = df[[feature, "noise_corrected_drift"]].dropna()
    if len(sub) < 3 or sub[feature].nunique() < 2 or sub["noise_corrected_drift"].nunique() < 2:
        return {
            "feature": feature,
            "n": len(sub),
            "spearman_rho": math.nan,
            "p_value": math.nan,
            "feature_family": "constant_or_missing",
        }
    rho, p_value = stats.spearmanr(sub[feature], sub["noise_corrected_drift"])
    return {
        "feature": feature,
        "n": len(sub),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "feature_family": "output" if feature in OUTPUT_FEATURES else "prompt",
    }


def run_regression(df: pd.DataFrame, name: str, predictors: list[str]) -> list[dict[str, object]]:
    if sm is None:
        return [
            {
                "model": name,
                "term": "ERROR",
                "n": 0,
                "r_squared": math.nan,
                "coef": math.nan,
                "std_error": math.nan,
                "p_value": math.nan,
                "note": f"statsmodels unavailable: {STATSMODELS_ERROR}",
            }
        ]

    cols = ["noise_corrected_drift", *predictors]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) <= len(predictors) + 2:
        return [
            {
                "model": name,
                "term": "ERROR",
                "n": len(sub),
                "r_squared": math.nan,
                "coef": math.nan,
                "std_error": math.nan,
                "p_value": math.nan,
                "note": "not enough complete rows",
            }
        ]

    y = sub["noise_corrected_drift"]
    x = sm.add_constant(sub[predictors], has_constant="add")
    fit = sm.OLS(y, x).fit(cov_type="HC3")
    rows = []
    for term in fit.params.index:
        rows.append(
            {
                "model": name,
                "term": term,
                "n": int(fit.nobs),
                "r_squared": float(fit.rsquared),
                "coef": float(fit.params[term]),
                "std_error": float(fit.bse[term]),
                "p_value": float(fit.pvalues[term]),
                "note": "",
            }
        )
    return rows


def build_feature_table(branch: str) -> pd.DataFrame:
    cfg = BRANCHES[branch]
    item = pd.read_csv(cfg["item"])
    cue = pd.read_csv(cfg["cue"])
    correctness = pd.read_csv(cfg["correctness"])
    original = pd.read_csv(cfg["original"])
    fixed = pd.read_csv(cfg["fixed"])

    base_cols = [
        "branch",
        "item_id",
        "reference_answer",
        "original_question",
        "paraphrased_question",
        "original_prompt",
        "perturbed_prompt",
        "noise_corrected_drift",
    ]
    cue_cols = [
        "item_id",
        "question_content_recall",
        "cue_disruption",
        "question_context_content_overlap_delta",
    ]
    correctness_cols = [
        "item_id",
        "original_mean_factual_score",
        "paraphrase_mean_factual_score",
        "factual_score_delta",
        "original_containment_rate",
        "paraphrase_containment_rate",
        "containment_rate_delta",
        "original_mean_output_length_tokens",
        "paraphrase_mean_output_length_tokens",
        "output_length_delta_tokens",
    ]

    df = (
        item[base_cols]
        .merge(cue[cue_cols], on="item_id", how="left")
        .merge(correctness[correctness_cols], on="item_id", how="left")
    )
    df = add_prompt_features(df)
    df = add_output_features(df, original, fixed)
    df["answer_compactness_loss_proxy"] = -zscore(df["factual_score_delta"])
    df["style_expansion_proxy"] = zscore(df["output_length_delta_tokens"])
    df["answer_scope_proxy"] = (
        zscore(df["output_length_delta_tokens"])
        - zscore(df["factual_score_delta"])
        + zscore(df["containment_rate_delta"])
    )
    return df


def make_summary(
    branch: str,
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    output_paths: dict[str, Path],
) -> str:
    cfg = BRANCHES[branch]
    top_positive = correlations.sort_values("spearman_rho", ascending=False).head(6)
    top_negative = correlations.sort_values("spearman_rho", ascending=True).head(6)

    def corr_table(rows: pd.DataFrame) -> str:
        lines = ["| Feature | rho | p | n |", "|---|---:|---:|---:|"]
        for _, row in rows.iterrows():
            lines.append(
                f"| `{row['feature']}` | {row['spearman_rho']:.4f} | {row['p_value']:.4g} | {int(row['n'])} |"
            )
        return "\n".join(lines)

    def regression_model_table(model_name: str) -> str:
        rows = regressions[regressions["model"] == model_name]
        if rows.empty:
            return "_No rows._"
        lines = ["| Term | coef | p | R2 |", "|---|---:|---:|---:|"]
        for _, row in rows.iterrows():
            if row["term"] == "const":
                continue
            lines.append(f"| `{row['term']}` | {row['coef']:.4f} | {row['p_value']:.4g} | {row['r_squared']:.4f} |")
        return "\n".join(lines)

    output_length_corr = correlations[correlations["feature"] == "output_length_delta_tokens"].iloc[0]
    f1_corr = correlations[correlations["feature"] == "factual_score_delta"].iloc[0]
    containment_corr = correlations[correlations["feature"] == "containment_rate_delta"].iloc[0]
    scope_corr = correlations[correlations["feature"] == "answer_scope_proxy"].iloc[0]
    question_edit_corr = correlations[correlations["feature"] == "question_token_edit_distance_norm"].iloc[0]

    output_feature_mean_abs = correlations[correlations["feature_family"] == "output"]["spearman_rho"].abs().mean()
    prompt_feature_mean_abs = correlations[correlations["feature_family"] == "prompt"]["spearman_rho"].abs().mean()

    lines = [
        f"# Factual Text-Level Feature Driver Analysis ({cfg['label']})",
        "",
        "Run date: 2026-07-10",
        "",
        "## Purpose",
        "",
        "This analysis tests whether observable text-level changes explain the residual fixed factual QA paraphrase drift. It is designed to strengthen or qualify the current scope/style explanation before moving to probability or attention probes.",
        "",
        "## Step 1. Feature Base Validation",
        "",
        f"- Rows: {len(df)}",
        f"- Unique items: {df['item_id'].nunique()}",
        f"- Missing `noise_corrected_drift`: {int(df['noise_corrected_drift'].isna().sum())}",
        f"- Missing original/paraphrased questions: {int(df[['original_question', 'paraphrased_question']].isna().any(axis=1).sum())}",
        f"- Mean fixed factual QA NCP: {df['noise_corrected_drift'].mean():.6f}",
        "",
        f"Conclusion: the `{branch}` fixed factual QA table is complete enough for text-level driver analysis.",
        "",
        "## Step 2. Prompt-Side Rewrite Features",
        "",
        f"- Mean normalized question token edit distance: {df['question_token_edit_distance_norm'].mean():.4f}",
        f"- Mean question length delta: {df['question_length_delta_tokens'].mean():.3f} tokens",
        f"- Mean normalized prompt token edit distance: {df['prompt_token_edit_distance_norm'].mean():.4f}",
        f"- `question_token_edit_distance_norm` vs drift: rho={question_edit_corr['spearman_rho']:.4f}, p={question_edit_corr['p_value']:.4g}",
        "",
        "Conclusion: prompt/question rewrite magnitude is measured directly. Its explanatory value should be judged against the output-side features below.",
        "",
        "## Step 3. Output-Side Edit And Expansion Features",
        "",
        f"- Mean output length delta: {df['output_length_delta_tokens'].mean():.3f} tokens",
        f"- Mean output length ratio: {df['output_length_ratio'].mean():.4f}",
        f"- Mean all-pairs normalized output token edit distance: {df['mean_output_token_edit_distance_norm'].mean():.4f}",
        f"- `output_length_delta_tokens` vs drift: rho={output_length_corr['spearman_rho']:.4f}, p={output_length_corr['p_value']:.4g}",
        "",
        "Conclusion: this step quantifies whether drift is visible as generated-output expansion and direct output text divergence.",
        "",
        "## Step 4. Scope / Style Proxy Features",
        "",
        f"- `factual_score_delta` vs drift: rho={f1_corr['spearman_rho']:.4f}, p={f1_corr['p_value']:.4g}",
        f"- `containment_rate_delta` vs drift: rho={containment_corr['spearman_rho']:.4f}, p={containment_corr['p_value']:.4g}",
        f"- `answer_scope_proxy` vs drift: rho={scope_corr['spearman_rho']:.4f}, p={scope_corr['p_value']:.4g}",
        "",
        "Conclusion: the proxy separates compact-reference-answer loss and output expansion from simple reference containment failure.",
        "",
        "## Step 5. Correlation Ranking",
        "",
        "Top positive relationships with drift:",
        "",
        corr_table(top_positive),
        "",
        "Top negative relationships with drift:",
        "",
        corr_table(top_negative),
        "",
        f"Mean absolute rho for output-side features: {output_feature_mean_abs:.4f}",
        f"Mean absolute rho for prompt-side features: {prompt_feature_mean_abs:.4f}",
        "",
        "Conclusion: compare these two means and the ranked features to decide whether generated-answer form explains more than prompt rewrite magnitude.",
        "",
        "## Step 6. Exploratory Regressions",
        "",
        "Minimal text-driver model:",
        "",
        regression_model_table("minimal_text_driver"),
        "",
        "Scope/style model:",
        "",
        regression_model_table("scope_style"),
        "",
        "Output-distance model:",
        "",
        regression_model_table("output_distance"),
        "",
        "Conclusion: these regressions are exploratory because n=50. Use coefficient direction, R2, and consistency with Spearman patterns rather than treating p-values as decisive.",
        "",
        "## Files Written",
        "",
        f"- `{output_paths['base'].relative_to(ROOT)}`",
        f"- `{output_paths['correlations'].relative_to(ROOT)}`",
        f"- `{output_paths['regressions'].relative_to(ROOT)}`",
        f"- `{output_paths['summary'].relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def run(branch: str) -> dict[str, Path]:
    cfg = BRANCHES[branch]
    out_dir = cfg["dir"]
    prefix = "" if branch == "outputs" else f"{branch}_"
    output_paths = {
        "base": out_dir / "factual_text_feature_base_fixed_factual.csv",
        "correlations": out_dir / "factual_text_feature_driver_correlations_fixed_factual.csv",
        "regressions": out_dir / "factual_text_feature_driver_regressions_fixed_factual.csv",
        "summary": out_dir / "factual_text_feature_driver_summary_fixed_factual.md",
    }

    df = build_feature_table(branch)
    features = PROMPT_FEATURES + OUTPUT_FEATURES
    correlations = pd.DataFrame([spearman(df, feature) for feature in features])
    correlations = correlations.sort_values(["spearman_rho"], ascending=False)
    regression_rows = []
    for name, predictors in REGRESSION_SPECS:
        regression_rows.extend(run_regression(df, name, predictors))
    regressions = pd.DataFrame(regression_rows)

    try:
        df.to_csv(output_paths["base"], index=False)
        correlations.to_csv(output_paths["correlations"], index=False)
        regressions.to_csv(output_paths["regressions"], index=False)
        output_paths["summary"].write_text(
            make_summary(branch, df, correlations, regressions, output_paths),
            encoding="utf-8",
        )
    except PermissionError:
        fallback_dir = ROOT / "outputs"
        output_paths = {
            "base": fallback_dir / f"{prefix}factual_text_feature_base_fixed_factual.csv",
            "correlations": fallback_dir / f"{prefix}factual_text_feature_driver_correlations_fixed_factual.csv",
            "regressions": fallback_dir / f"{prefix}factual_text_feature_driver_regressions_fixed_factual.csv",
            "summary": fallback_dir / f"{prefix}factual_text_feature_driver_summary_fixed_factual.md",
        }
        df.to_csv(output_paths["base"], index=False)
        correlations.to_csv(output_paths["correlations"], index=False)
        regressions.to_csv(output_paths["regressions"], index=False)
        output_paths["summary"].write_text(
            make_summary(branch, df, correlations, regressions, output_paths),
            encoding="utf-8",
        )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=sorted(BRANCHES), default="outputs")
    args = parser.parse_args()
    paths = run(args.branch)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
