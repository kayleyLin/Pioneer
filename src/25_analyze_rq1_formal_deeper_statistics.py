"""Deeper statistical checks for formal RQ1.

This script adds three analyses that are useful when heatmap trends are not
visually clean:

1. Bootstrap confidence intervals for each task x perturbation cell.
2. One-sample tests asking whether corrected drift is greater than zero.
3. Bootstrap rank stability showing how often each perturbation ranks highest
   within each task type.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "sbert_rq1_formal_perturbation_effects_by_item.csv"
CELL_STATS = ROOT / "outputs" / "rq1_formal_deeper_cell_statistics.csv"
RANK_STABILITY = ROOT / "outputs" / "rq1_formal_deeper_rank_stability.csv"

SEED = 20260702
N_BOOTSTRAPS = 10000
PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = []
    for _ in range(N_BOOTSTRAPS):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def cell_statistics(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    p_values = []

    for (task_type, perturbation_type), group in sorted(
        df.groupby(["task_type", "perturbation_type"])
    ):
        values = group["noise_corrected_drift"].to_numpy(dtype=float)
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci_lower, ci_upper = bootstrap_ci(values, rng)

        # One-sided Wilcoxon: is corrected drift greater than zero?
        try:
            wilcoxon = stats.wilcoxon(values, alternative="greater")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0

        ttest = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
        cohen_d = mean_value / std_value if std_value else 0.0

        row = {
            "task_type": task_type,
            "perturbation_type": perturbation_type,
            "n_items": len(values),
            "mean_noise_corrected_drift": round(mean_value, 6),
            "std_noise_corrected_drift": round(std_value, 6),
            "bootstrap_ci_lower": round(ci_lower, 6),
            "bootstrap_ci_upper": round(ci_upper, 6),
            "ci_excludes_zero": ci_lower > 0 or ci_upper < 0,
            "wilcoxon_greater_statistic": round(wilcoxon_stat, 6),
            "wilcoxon_greater_p": round(wilcoxon_p, 6),
            "ttest_greater_p": round(float(ttest.pvalue), 6),
            "cohen_d_vs_zero": round(cohen_d, 6),
        }
        rows.append(row)
        p_values.append(wilcoxon_p)

    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    for row, is_reject, adjusted in zip(rows, reject, p_adj):
        row["wilcoxon_holm_p"] = round(float(adjusted), 6)
        row["wilcoxon_holm_reject_0_05"] = bool(is_reject)

    write_rows(CELL_STATS, rows)


def rank_stability(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    for task_type, task_df in sorted(df.groupby("task_type")):
        wide = task_df.pivot(
            index="item_id",
            columns="perturbation_type",
            values="noise_corrected_drift",
        )[PERTURBATION_ORDER]

        top_counts: dict[str, int] = defaultdict(int)
        for _ in range(N_BOOTSTRAPS):
            sampled_index = rng.choice(wide.index.to_numpy(), size=len(wide), replace=True)
            sampled = wide.loc[sampled_index]
            means = sampled.mean(axis=0)
            top_counts[str(means.idxmax())] += 1

        for perturbation_type in PERTURBATION_ORDER:
            rows.append(
                {
                    "task_type": task_type,
                    "perturbation_type": perturbation_type,
                    "top_rank_probability": round(
                        top_counts[perturbation_type] / N_BOOTSTRAPS, 6
                    ),
                    "n_bootstraps": N_BOOTSTRAPS,
                }
            )

    write_rows(RANK_STABILITY, rows)


def main() -> None:
    df = pd.read_csv(INPUT)
    df["noise_corrected_drift"] = df["noise_corrected_drift"].astype(float)
    cell_statistics(df)
    rank_stability(df)
    print(f"Wrote {CELL_STATS}")
    print(f"Wrote {RANK_STABILITY}")


if __name__ == "__main__":
    main()
