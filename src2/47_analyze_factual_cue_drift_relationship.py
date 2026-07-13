"""Analyze fixed factual-QA paraphrase drift relationships.

Formal Step 7 script. Spearman correlations are primary. OLS regressions are
exploratory and use HC3 robust standard errors when statsmodels is available.

Outputs:
    outputs/factual_paraphrase_cue_correlations_fixed_factual.csv
    outputs/factual_paraphrase_cue_regressions_fixed_factual.csv
    outputs/factual_paraphrase_cue_analysis_fixed_factual.md
    qwen/outputs/...
    llama/outputs/...
"""

from __future__ import annotations

import argparse
import math
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
        "input": ROOT / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "correlations": ROOT / "outputs" / "factual_paraphrase_cue_correlations_fixed_factual.csv",
        "regressions": ROOT / "outputs" / "factual_paraphrase_cue_regressions_fixed_factual.csv",
        "analysis": ROOT / "outputs" / "factual_paraphrase_cue_analysis_fixed_factual.md",
    },
    "qwen": {
        "input": ROOT / "qwen" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "correlations": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_correlations_fixed_factual.csv",
        "regressions": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_regressions_fixed_factual.csv",
        "analysis": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_analysis_fixed_factual.md",
    },
    "llama": {
        "input": ROOT / "llama" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "correlations": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_correlations_fixed_factual.csv",
        "regressions": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_regressions_fixed_factual.csv",
        "analysis": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_analysis_fixed_factual.md",
    },
}

CORRELATION_SPECS = [
    ("cue_disruption", "noise_corrected_drift"),
    ("critical_cue_preservation", "noise_corrected_drift"),
    ("question_content_recall", "noise_corrected_drift"),
    ("capitalized_phrase_recall", "noise_corrected_drift"),
    ("wh_word_preserved", "noise_corrected_drift"),
    ("question_context_content_overlap_delta", "noise_corrected_drift"),
    ("question_context_content_overlap_loss", "noise_corrected_drift"),
    ("factual_score_delta", "noise_corrected_drift"),
    ("containment_rate_delta", "noise_corrected_drift"),
    ("output_length_delta_tokens", "noise_corrected_drift"),
    ("cue_disruption", "factual_score_delta"),
    ("question_content_recall", "factual_score_delta"),
]

REGRESSION_SPECS = [
    ("cue_only", ["cue_disruption"]),
    ("correctness_only", ["factual_score_delta"]),
    ("length_only", ["output_length_delta_tokens"]),
    ("cue_correctness_length", ["cue_disruption", "factual_score_delta", "output_length_delta_tokens"]),
    (
        "expanded",
        [
            "cue_disruption",
            "question_content_recall",
            "question_context_content_overlap_loss",
            "factual_score_delta",
            "output_length_delta_tokens",
        ],
    ),
]


def spearman(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, object]:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 3 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return {
            "x": x_col,
            "y": y_col,
            "n": len(sub),
            "spearman_rho": math.nan,
            "p_value": math.nan,
        }
    rho, p_value = stats.spearmanr(sub[x_col], sub[y_col])
    return {
        "x": x_col,
        "y": y_col,
        "n": len(sub),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
    }


def regression(df: pd.DataFrame, name: str, predictors: list[str]) -> list[dict[str, object]]:
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
    sub = df[cols].dropna()
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
                "note": "not enough rows",
            }
        ]
    x = sm.add_constant(sub[predictors], has_constant="add")
    y = sub["noise_corrected_drift"]
    result = sm.OLS(y, x).fit(cov_type="HC3")
    rows = []
    for term in result.params.index:
        rows.append(
            {
                "model": name,
                "term": term,
                "n": int(result.nobs),
                "r_squared": float(result.rsquared),
                "coef": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "p_value": float(result.pvalues[term]),
                "note": "OLS with HC3 robust SE",
            }
        )
    return rows


def validate(branch: str, correlations: pd.DataFrame, regressions: pd.DataFrame) -> None:
    failures = []
    if correlations.empty:
        failures.append("correlations table is empty")
    if regressions.empty:
        failures.append("regressions table is empty")
    if not {"x", "y", "spearman_rho", "p_value"}.issubset(correlations.columns):
        failures.append("correlations table missing required columns")
    if not {"model", "term", "coef", "p_value"}.issubset(regressions.columns):
        failures.append("regressions table missing required columns")
    if failures:
        raise SystemExit(f"{branch}: validation failed: " + "; ".join(failures))


def write_markdown(branch: str, df: pd.DataFrame, correlations: pd.DataFrame, regressions: pd.DataFrame, path: Path) -> None:
    primary = correlations[correlations["y"] == "noise_corrected_drift"].copy()
    primary["abs_rho"] = primary["spearman_rho"].abs()
    primary = primary.sort_values("abs_rho", ascending=False).drop(columns=["abs_rho"])

    selected_models = regressions[
        regressions["model"].isin(["cue_correctness_length", "expanded"])
        & (regressions["term"] != "const")
    ].copy()

    lines = [
        f"# Fixed Factual QA Paraphrase Drift Analysis: {branch}",
        "",
        f"- Rows: {len(df)}",
        f"- Mean noise-corrected drift: {df['noise_corrected_drift'].mean():.6f}",
        f"- Mean cue disruption: {df['cue_disruption'].mean():.6f}",
        f"- Mean factual score delta: {df['factual_score_delta'].mean():.6f}",
        f"- Mean output length delta tokens: {df['output_length_delta_tokens'].mean():.6f}",
        "",
        "## Primary Spearman Correlations With Drift",
        "",
        primary.to_markdown(index=False),
        "",
        "## Selected Exploratory OLS Terms",
        "",
        selected_models.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "Spearman correlations are the primary evidence because n=50 is small. "
            "OLS models are exploratory and are included to check whether cue, "
            "correctness, and output-length signals survive in the same model."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_branch(branch: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = BRANCHES[branch]
    df = pd.read_csv(config["input"])
    correlations = pd.DataFrame([spearman(df, x, y) for x, y in CORRELATION_SPECS])
    regression_rows = []
    for name, predictors in REGRESSION_SPECS:
        regression_rows.extend(regression(df, name, predictors))
    regressions = pd.DataFrame(regression_rows)
    validate(branch, correlations, regressions)
    config["correlations"].parent.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(config["correlations"], index=False)
    regressions.to_csv(config["regressions"], index=False)
    write_markdown(branch, df, correlations, regressions, config["analysis"])
    print(f"Wrote {config['correlations'].relative_to(ROOT)}")
    print(f"Wrote {config['regressions'].relative_to(ROOT)}")
    print(f"Wrote {config['analysis'].relative_to(ROOT)}")
    top = correlations[correlations["y"] == "noise_corrected_drift"].copy()
    top["abs_rho"] = top["spearman_rho"].abs()
    top = top.sort_values("abs_rho", ascending=False).head(3)
    print(branch)
    print(top[["x", "spearman_rho", "p_value", "n"]].to_string(index=False))
    return correlations, regressions


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
