"""Summarize the position of factual-QA paraphrasing in RQ1 n=50 results.

Inputs:
    outputs/sbert_rq1_n50_perturbation_summary.csv
    qwen/outputs/sbert_rq1_n50_perturbation_summary.csv
    llama/outputs/sbert_rq1_n50_perturbation_summary.csv

Outputs:
    outputs/factual_paraphrase_position_summary.csv
    outputs/factual_paraphrase_position_summary.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

TARGET_TASK = "factual_qa"
TARGET_PERTURBATION = "paraphrasing"

BRANCHES = [
    {
        "branch": "outputs",
        "model_label": "ChatGPT GPT-4o mini",
        "summary_path": ROOT / "outputs" / "sbert_rq1_n50_perturbation_summary.csv",
    },
    {
        "branch": "qwen",
        "model_label": "Qwen",
        "summary_path": ROOT / "qwen" / "outputs" / "sbert_rq1_n50_perturbation_summary.csv",
    },
    {
        "branch": "llama",
        "model_label": "Llama",
        "summary_path": ROOT / "llama" / "outputs" / "sbert_rq1_n50_perturbation_summary.csv",
    },
]


def format_cell(row: pd.Series) -> str:
    return f"{row['task_type']} + {row['perturbation_type']}"


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    df = pd.read_csv(path)
    required = {
        "task_type",
        "perturbation_type",
        "n_items",
        "mean_noise_corrected_drift",
        "std_noise_corrected_drift",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    if len(df) != 20:
        raise SystemExit(f"{path} should have 20 rows, found {len(df)}")
    return df


def summarize_branch(config: dict[str, object]) -> dict[str, object]:
    branch = str(config["branch"])
    model_label = str(config["model_label"])
    path = config["summary_path"]
    assert isinstance(path, Path)

    df = load_summary(path).copy()
    df = df.sort_values("mean_noise_corrected_drift", ascending=False).reset_index(drop=True)
    df["rank_descending"] = df["mean_noise_corrected_drift"].rank(
        method="min", ascending=False
    ).astype(int)

    target_rows = df[
        (df["task_type"] == TARGET_TASK)
        & (df["perturbation_type"] == TARGET_PERTURBATION)
    ]
    if len(target_rows) != 1:
        raise SystemExit(f"{path} should contain exactly one factual_qa + paraphrasing row.")

    target = target_rows.iloc[0]
    other = df.drop(index=target.name)
    next_largest = other.iloc[0]
    non_factual_paraphrase = df[
        (df["task_type"] != TARGET_TASK)
        & (df["perturbation_type"] == TARGET_PERTURBATION)
    ]
    other_factual = df[
        (df["task_type"] == TARGET_TASK)
        & (df["perturbation_type"] != TARGET_PERTURBATION)
    ]

    target_mean = float(target["mean_noise_corrected_drift"])
    next_mean = float(next_largest["mean_noise_corrected_drift"])
    non_factual_paraphrase_mean = float(non_factual_paraphrase["mean_noise_corrected_drift"].mean())
    other_factual_mean = float(other_factual["mean_noise_corrected_drift"].mean())

    return {
        "branch": branch,
        "model_label": model_label,
        "target_cell": format_cell(target),
        "target_rank_descending": int(target["rank_descending"]),
        "target_mean_noise_corrected_drift": target_mean,
        "target_std_noise_corrected_drift": float(target["std_noise_corrected_drift"]),
        "target_n_items": int(target["n_items"]),
        "next_largest_cell": format_cell(next_largest),
        "next_largest_mean_noise_corrected_drift": next_mean,
        "gap_from_next_largest": target_mean - next_mean,
        "mean_non_factual_paraphrasing_drift": non_factual_paraphrase_mean,
        "ratio_to_non_factual_paraphrasing_mean": target_mean / non_factual_paraphrase_mean,
        "mean_other_factual_qa_perturbation_drift": other_factual_mean,
        "ratio_to_other_factual_qa_perturbation_mean": target_mean / other_factual_mean,
        "passes_rank_1_check": int(target["rank_descending"]) == 1,
        "input_file": str(path.relative_to(ROOT)),
    }


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    rank_failures = summary[~summary["passes_rank_1_check"]]
    lines = [
        "# Factual QA Paraphrasing Position Summary",
        "",
        "This table checks whether `factual_qa + paraphrasing` is the largest noise-corrected drift cell in each available branch.",
        "",
        f"- Branches checked: {len(summary)}",
        f"- Branches where target ranks first: {int(summary['passes_rank_1_check'].sum())}",
        f"- Branches where target does not rank first: {len(rank_failures)}",
        "",
    ]
    if rank_failures.empty:
        lines.extend(
            [
                "## Conclusion",
                "",
                "PASS. `factual_qa + paraphrasing` ranks first in ChatGPT/main, Qwen, and Llama summaries.",
                "",
            ]
        )
    else:
        lines.extend(["## Rank Check Failures", "", rank_failures.to_markdown(index=False), ""])

    display_cols = [
        "branch",
        "model_label",
        "target_rank_descending",
        "target_mean_noise_corrected_drift",
        "next_largest_cell",
        "next_largest_mean_noise_corrected_drift",
        "gap_from_next_largest",
        "ratio_to_non_factual_paraphrasing_mean",
        "ratio_to_other_factual_qa_perturbation_mean",
    ]
    lines.extend(["## Summary Table", "", summary[display_cols].to_markdown(index=False), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = [summarize_branch(config) for config in BRANCHES]
    summary = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS / "factual_paraphrase_position_summary.csv"
    md_path = OUTPUTS / "factual_paraphrase_position_summary.md"
    summary.to_csv(csv_path, index=False)
    write_markdown(summary, md_path)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    if not summary["passes_rank_1_check"].all():
        raise SystemExit("Rank check failed for at least one branch.")
    print("All branches passed factual_qa + paraphrasing rank-1 check.")


if __name__ == "__main__":
    main()
