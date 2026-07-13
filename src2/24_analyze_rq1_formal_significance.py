"""Run statistical checks for formal RQ1 results.

This script focuses on two questions:

1. Do baseline sampling-noise drifts differ across task types?
   - Independent groups: one-way ANOVA, Kruskal-Wallis, Tukey HSD.

2. Within each task type, do noise-corrected perturbation drifts differ across
   perturbation types?
   - Repeated-measures design: Friedman test.
   - If needed, use paired Wilcoxon signed-rank tests with Holm correction.
"""

import csv
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs" / "sbert_rq1_formal_baseline_by_item.csv"
EFFECTS = ROOT / "outputs" / "sbert_rq1_formal_perturbation_effects_by_item.csv"

BASELINE_TESTS = ROOT / "outputs" / "rq1_formal_significance_baseline_tests.csv"
BASELINE_TUKEY = ROOT / "outputs" / "rq1_formal_significance_baseline_tukey.csv"
PERTURBATION_TESTS = (
    ROOT / "outputs" / "rq1_formal_significance_perturbation_friedman.csv"
)
PERTURBATION_PAIRWISE = (
    ROOT / "outputs" / "rq1_formal_significance_perturbation_pairwise_wilcoxon.csv"
)

PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def baseline_tests() -> None:
    df = pd.read_csv(BASELINE)
    df["sampling_noise_drift"] = df["sampling_noise_drift"].astype(float)

    groups = [
        group["sampling_noise_drift"].to_numpy()
        for _, group in df.groupby("task_type")
    ]

    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    levene = stats.levene(*groups, center="median")

    write_rows(
        BASELINE_TESTS,
        [
            {
                "test": "one_way_anova",
                "statistic": round(float(anova.statistic), 6),
                "p_value": round(float(anova.pvalue), 6),
                "interpretation": "tests mean baseline drift differences across task types",
            },
            {
                "test": "kruskal_wallis",
                "statistic": round(float(kruskal.statistic), 6),
                "p_value": round(float(kruskal.pvalue), 6),
                "interpretation": "nonparametric check for baseline drift differences across task types",
            },
            {
                "test": "levene_median",
                "statistic": round(float(levene.statistic), 6),
                "p_value": round(float(levene.pvalue), 6),
                "interpretation": "checks equality of variance across task types",
            },
        ],
    )

    tukey = pairwise_tukeyhsd(
        endog=df["sampling_noise_drift"],
        groups=df["task_type"],
        alpha=0.05,
    )
    tukey_rows = []
    for row in tukey.summary().data[1:]:
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        tukey_rows.append(
            {
                "group1": group1,
                "group2": group2,
                "mean_difference": round(float(meandiff), 6),
                "p_adj": round(float(p_adj), 6),
                "ci_lower": round(float(lower), 6),
                "ci_upper": round(float(upper), 6),
                "reject_alpha_0_05": bool(reject),
            }
        )
    write_rows(BASELINE_TUKEY, tukey_rows)


def perturbation_tests() -> None:
    df = pd.read_csv(EFFECTS)
    df["noise_corrected_drift"] = df["noise_corrected_drift"].astype(float)

    friedman_rows = []
    pairwise_rows = []

    for task_type, task_df in df.groupby("task_type"):
        wide = task_df.pivot(
            index="item_id",
            columns="perturbation_type",
            values="noise_corrected_drift",
        )[PERTURBATION_ORDER]

        friedman = stats.friedmanchisquare(
            *[wide[column].to_numpy() for column in PERTURBATION_ORDER]
        )
        friedman_rows.append(
            {
                "task_type": task_type,
                "test": "friedman_repeated_measures",
                "n_items": len(wide),
                "statistic": round(float(friedman.statistic), 6),
                "p_value": round(float(friedman.pvalue), 6),
            }
        )

        raw_pairwise = []
        for first, second in combinations(PERTURBATION_ORDER, 2):
            try:
                wilcoxon = stats.wilcoxon(wide[first], wide[second])
                statistic = float(wilcoxon.statistic)
                p_value = float(wilcoxon.pvalue)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
            raw_pairwise.append((first, second, statistic, p_value))

        adjusted = multipletests(
            [row[3] for row in raw_pairwise], alpha=0.05, method="holm"
        )
        reject_flags, p_adjusted = adjusted[0], adjusted[1]

        for (first, second, statistic, p_value), reject, p_adj in zip(
            raw_pairwise, reject_flags, p_adjusted
        ):
            pairwise_rows.append(
                {
                    "task_type": task_type,
                    "test": "wilcoxon_signed_rank_holm",
                    "perturbation_1": first,
                    "perturbation_2": second,
                    "n_items": len(wide),
                    "statistic": round(statistic, 6),
                    "p_value": round(p_value, 6),
                    "p_adj_holm": round(float(p_adj), 6),
                    "reject_alpha_0_05": bool(reject),
                }
            )

    write_rows(PERTURBATION_TESTS, friedman_rows)
    write_rows(PERTURBATION_PAIRWISE, pairwise_rows)


def main() -> None:
    baseline_tests()
    perturbation_tests()
    print(f"Wrote {BASELINE_TESTS}")
    print(f"Wrote {BASELINE_TUKEY}")
    print(f"Wrote {PERTURBATION_TESTS}")
    print(f"Wrote {PERTURBATION_PAIRWISE}")


if __name__ == "__main__":
    main()
