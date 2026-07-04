"""Run statistical tests for expanded RQ1 n=50 perturbation effects."""

import csv
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item.csv"
FRIEDMAN = ROOT / "outputs" / "rq1_n50_perturbation_friedman.csv"
PAIRWISE = ROOT / "outputs" / "rq1_n50_perturbation_pairwise_wilcoxon.csv"

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


def main() -> None:
    if not EFFECTS.exists():
        raise SystemExit(f"Missing input file: {EFFECTS}")

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
                "p_value": round(float(friedman.pvalue), 12),
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

        reject_flags, p_adjusted, _, _ = multipletests(
            [row[3] for row in raw_pairwise], alpha=0.05, method="holm"
        )

        for (first, second, statistic, p_value), reject, p_adj in zip(
            raw_pairwise, reject_flags, p_adjusted
        ):
            median_diff = float((wide[first] - wide[second]).median())
            mean_diff = float((wide[first] - wide[second]).mean())
            pairwise_rows.append(
                {
                    "task_type": task_type,
                    "test": "wilcoxon_signed_rank_holm",
                    "perturbation_1": first,
                    "perturbation_2": second,
                    "n_items": len(wide),
                    "mean_difference": round(mean_diff, 6),
                    "median_difference": round(median_diff, 6),
                    "statistic": round(statistic, 6),
                    "p_value": round(p_value, 12),
                    "p_adj_holm": round(float(p_adj), 12),
                    "reject_alpha_0_05": bool(reject),
                }
            )

    write_rows(FRIEDMAN, friedman_rows)
    write_rows(PAIRWISE, pairwise_rows)
    print(f"Wrote {FRIEDMAN}")
    print(f"Wrote {PAIRWISE}")


if __name__ == "__main__":
    main()
