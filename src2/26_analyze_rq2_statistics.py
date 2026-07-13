"""Run statistical analyses for RQ2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "rq2" / "outputs" / "rq2_formal_available_drift_performance_by_item.csv"
OUT_DIR = ROOT / "rq2" / "outputs"


def ci95(series: pd.Series) -> tuple[float, float]:
    values = series.dropna()
    if len(values) < 2:
        return (float("nan"), float("nan"))
    sem = stats.sem(values)
    margin = stats.t.ppf(0.975, len(values) - 1) * sem
    return (values.mean() - margin, values.mean() + margin)


def descriptive(df: pd.DataFrame, group_cols: list[str], output: Path) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(group)
        for col in ["noise_corrected_drift", "absolute_performance_change", "pdr"]:
            values = group[col].dropna()
            low, high = ci95(values)
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_sd"] = values.std(ddof=1)
            row[f"{col}_ci95_low"] = low
            row[f"{col}_ci95_high"] = high
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False)
    return out


def correlation_rows(df: pd.DataFrame, group_cols: list[str] | None = None) -> list[dict]:
    if group_cols is None:
        groups = [("overall", df)]
        label_cols = ["group"]
    else:
        groups = list(df.groupby(group_cols, dropna=False))
        label_cols = group_cols

    rows = []
    for keys, group in groups:
        if group_cols is None:
            labels = {"group": keys}
        else:
            if not isinstance(keys, tuple):
                keys = (keys,)
            labels = dict(zip(label_cols, keys))

        x = group["noise_corrected_drift"]
        y = group["absolute_performance_change"]
        valid = pd.concat([x, y], axis=1).dropna()
        row = {**labels, "n": len(valid)}
        if len(valid) >= 3:
            pearson = stats.pearsonr(valid["noise_corrected_drift"], valid["absolute_performance_change"])
            spearman = stats.spearmanr(valid["noise_corrected_drift"], valid["absolute_performance_change"])
            row.update(
                {
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_rho": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
        else:
            row.update(
                {
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                }
            )
        rows.append(row)
    return rows


def regression_summary_text(model, title: str) -> str:
    return f"\n\n## {title}\n\n```text\n{model.summary()}\n```\n"


def main() -> None:
    df = pd.read_csv(INPUT)
    numeric_cols = [
        "original_performance",
        "perturbed_performance",
        "absolute_performance_change",
        "pdr",
        "baseline_similarity",
        "perturbation_similarity",
        "noise_corrected_drift",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    descriptive(df, ["task_type"], OUT_DIR / "rq2_stats_descriptive_by_task.csv")
    descriptive(df, ["perturbation_type"], OUT_DIR / "rq2_stats_descriptive_by_perturbation.csv")
    descriptive(df, ["task_type", "perturbation_type"], OUT_DIR / "rq2_stats_descriptive_by_task_perturbation.csv")

    corr_rows = []
    corr_rows.extend(correlation_rows(df, None))
    corr_rows.extend(correlation_rows(df, ["task_type"]))
    corr_rows.extend(correlation_rows(df, ["perturbation_type"]))
    corr_rows.extend(correlation_rows(df, ["task_type", "perturbation_type"]))
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "rq2_stats_correlations.csv", index=False)

    model_1 = smf.ols(
        "absolute_performance_change ~ noise_corrected_drift",
        data=df,
    ).fit(cov_type="HC3")
    model_2 = smf.ols(
        "absolute_performance_change ~ noise_corrected_drift * C(task_type)",
        data=df,
    ).fit(cov_type="HC3")
    model_3 = smf.ols(
        "absolute_performance_change ~ noise_corrected_drift * C(task_type) + C(perturbation_type)",
        data=df,
    ).fit(cov_type="HC3")

    mixed_text = ""
    try:
        mixed = smf.mixedlm(
            "absolute_performance_change ~ noise_corrected_drift * C(task_type) + C(perturbation_type)",
            data=df,
            groups=df["item_id"],
        ).fit(reml=False, method="lbfgs")
        mixed_text = regression_summary_text(mixed, "Mixed-Effects Robustness Check")
    except Exception as error:
        mixed_text = (
            "\n\n## Mixed-Effects Robustness Check\n\n"
            f"Mixed-effects model did not converge or could not be fit: `{type(error).__name__}: {error}`\n"
        )

    report = [
        "# RQ2 Statistical Analysis\n",
        "## Input\n\n",
        f"Input table: `{INPUT}`\n\n",
        "Rows: 150 item-perturbation observations.\n\n",
        "Outcome: `absolute_performance_change`.\n\n",
        "Main predictor: `noise_corrected_drift`.\n\n",
        "Task and perturbation type are treated as categorical factors.\n",
        regression_summary_text(model_1, "OLS 1: Drift Only"),
        regression_summary_text(model_2, "OLS 2: Drift By Task Interaction"),
        regression_summary_text(model_3, "OLS 3: Drift By Task Interaction + Perturbation Controls"),
        mixed_text,
        "\n## Output CSV Files\n\n",
        "```text\n",
        "rq2/outputs/rq2_stats_descriptive_by_task.csv\n",
        "rq2/outputs/rq2_stats_descriptive_by_perturbation.csv\n",
        "rq2/outputs/rq2_stats_descriptive_by_task_perturbation.csv\n",
        "rq2/outputs/rq2_stats_correlations.csv\n",
        "```\n",
    ]
    (ROOT / "rq2" / "rq2_statistical_analysis.md").write_text("".join(report), encoding="utf-8")
    print("Wrote RQ2 statistical analysis outputs.")


if __name__ == "__main__":
    main()
