"""Select qualitative cases for fixed factual-QA paraphrase discussion.

Step 8 script. It selects candidate cases that illustrate high drift, low drift,
cue disruption, answer-scope expansion, output-length changes, and factual-score
changes under the repaired context-preserving factual QA paraphrase condition.

Outputs:
    outputs/factual_paraphrase_case_candidates_fixed_factual.csv
    outputs/factual_paraphrase_case_table_fixed_factual.md
    outputs/factual_paraphrase_case_table_manual_review_fixed_factual.md
    qwen/outputs/...
    llama/outputs/...
    outputs/factual_paraphrase_case_candidates_cross_model_fixed_factual.csv
    outputs/factual_paraphrase_case_table_cross_model_fixed_factual.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BRANCHES = {
    "outputs": {
        "input": ROOT / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "output_dir": ROOT / "outputs",
    },
    "qwen": {
        "input": ROOT / "qwen" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "output_dir": ROOT / "qwen" / "outputs",
    },
    "llama": {
        "input": ROOT / "llama" / "outputs" / "factual_paraphrase_correctness_by_item_fixed_factual.csv",
        "output_dir": ROOT / "llama" / "outputs",
    },
}


def add_case(
    selected: dict[str, dict[str, object]],
    row: pd.Series,
    category: str,
    interpretation: str,
) -> None:
    item_id = str(row["item_id"])
    if item_id in selected:
        selected[item_id]["selection_category"] += f"; {category}"
        selected[item_id]["interpretation"] += f" {interpretation}"
        return
    record = row.to_dict()
    record["selection_category"] = category
    record["interpretation"] = interpretation
    selected[item_id] = record


def branch_cases(df: pd.DataFrame) -> pd.DataFrame:
    selected: dict[str, dict[str, object]] = {}

    for _, row in df.sort_values("noise_corrected_drift", ascending=False).head(3).iterrows():
        add_case(
            selected,
            row,
            "high_drift",
            "High fixed paraphrase drift; useful for inspecting output scope and wording changes.",
        )

    for _, row in df.sort_values("noise_corrected_drift", ascending=True).head(2).iterrows():
        add_case(
            selected,
            row,
            "low_drift",
            "Low fixed paraphrase drift; useful contrast case where paraphrase remains behaviorally stable.",
        )

    cue_subset = df.sort_values(["cue_disruption", "noise_corrected_drift"], ascending=[False, False]).head(2)
    for _, row in cue_subset.iterrows():
        add_case(
            selected,
            row,
            "cue_disruption",
            "High cue disruption; inspect WH/content cue or answer-type changes.",
        )

    length_subset = df.sort_values("output_length_delta_tokens", ascending=False).head(2)
    for _, row in length_subset.iterrows():
        add_case(
            selected,
            row,
            "output_length_increase",
            "Large paraphrase output-length increase; likely response-style or answer-scope expansion.",
        )

    f1_subset = df.sort_values("factual_score_delta", ascending=True).head(2)
    for _, row in f1_subset.iterrows():
        add_case(
            selected,
            row,
            "factual_score_drop",
            "Large token-F1 drop against reference; inspect whether answer is wrong or less compact.",
        )

    cases = pd.DataFrame(selected.values())
    cases = cases.sort_values(
        ["noise_corrected_drift", "cue_disruption"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return cases


def compact_case_columns(cases: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "item_id",
        "selection_category",
        "reference_answer",
        "original_question",
        "paraphrased_question",
        "noise_corrected_drift",
        "cue_disruption",
        "question_content_recall",
        "factual_score_delta",
        "containment_rate_delta",
        "output_length_delta_tokens",
        "interpretation",
    ]
    return cases[cols].copy()


def write_branch_outputs(branch: str, cases: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "factual_paraphrase_case_candidates_fixed_factual.csv"
    md_path = output_dir / "factual_paraphrase_case_table_fixed_factual.md"
    review_path = output_dir / "factual_paraphrase_case_table_manual_review_fixed_factual.md"

    compact = compact_case_columns(cases)
    compact.to_csv(csv_path, index=False)

    lines = [
        f"# Fixed Factual QA Paraphrase Case Table: {branch}",
        "",
        compact.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    review = compact.copy()
    review["answer_type_preserved"] = ""
    review["relation_cue_changed"] = ""
    review["answer_scope_changed"] = ""
    review["final_include_exclude"] = ""
    review["manual_notes"] = ""
    review_lines = [
        f"# Manual Review: Fixed Factual QA Paraphrase Cases ({branch})",
        "",
        review.to_markdown(index=False),
        "",
    ]
    review_path.write_text("\n".join(review_lines), encoding="utf-8")

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"Wrote {review_path.relative_to(ROOT)}")


def cross_model_cases() -> pd.DataFrame:
    frames = []
    for branch, config in BRANCHES.items():
        df = pd.read_csv(config["input"])
        df["model_branch"] = branch
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)

    grouped = (
        all_rows.groupby("item_id")
        .agg(
            reference_answer=("reference_answer", "first"),
            original_question=("original_question", "first"),
            paraphrased_question=("paraphrased_question", "first"),
            mean_noise_corrected_drift=("noise_corrected_drift", "mean"),
            max_noise_corrected_drift=("noise_corrected_drift", "max"),
            mean_cue_disruption=("cue_disruption", "mean"),
            mean_factual_score_delta=("factual_score_delta", "mean"),
            mean_output_length_delta_tokens=("output_length_delta_tokens", "mean"),
            branches_high_drift=("noise_corrected_drift", lambda values: int((values >= 0.15).sum())),
        )
        .reset_index()
    )

    selected: dict[str, dict[str, object]] = {}
    for _, row in grouped.sort_values("mean_noise_corrected_drift", ascending=False).head(4).iterrows():
        add_case(
            selected,
            row,
            "cross_model_high_mean_drift",
            "High average drift across model branches.",
        )
    for _, row in grouped.sort_values("mean_output_length_delta_tokens", ascending=False).head(2).iterrows():
        add_case(
            selected,
            row,
            "cross_model_length_increase",
            "Large average output-length increase across branches.",
        )
    for _, row in grouped.sort_values("mean_cue_disruption", ascending=False).head(2).iterrows():
        add_case(
            selected,
            row,
            "cross_model_cue_disruption",
            "High prompt-level cue disruption across branches.",
        )

    cases = pd.DataFrame(selected.values())
    return cases.sort_values("mean_noise_corrected_drift", ascending=False).reset_index(drop=True)


def write_cross_model(cases: pd.DataFrame) -> None:
    csv_path = ROOT / "outputs" / "factual_paraphrase_case_candidates_cross_model_fixed_factual.csv"
    md_path = ROOT / "outputs" / "factual_paraphrase_case_table_cross_model_fixed_factual.md"
    cols = [
        "item_id",
        "selection_category",
        "reference_answer",
        "original_question",
        "paraphrased_question",
        "mean_noise_corrected_drift",
        "max_noise_corrected_drift",
        "mean_cue_disruption",
        "mean_factual_score_delta",
        "mean_output_length_delta_tokens",
        "branches_high_drift",
        "interpretation",
    ]
    compact = cases[cols].copy()
    compact.to_csv(csv_path, index=False)
    lines = [
        "# Cross-Model Fixed Factual QA Paraphrase Case Table",
        "",
        compact.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


def validate_cases(branch: str, cases: pd.DataFrame) -> None:
    failures = []
    if len(cases) < 6:
        failures.append(f"expected at least 6 cases, found {len(cases)}")
    if cases["item_id"].duplicated().any():
        failures.append("duplicate item_id values")
    if cases["selection_category"].isna().any():
        failures.append("missing selection categories")
    if failures:
        raise SystemExit(f"{branch}: validation failed: " + "; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["outputs", "qwen", "llama", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    for branch in branches:
        config = BRANCHES[branch]
        df = pd.read_csv(config["input"])
        cases = branch_cases(df)
        validate_cases(branch, cases)
        write_branch_outputs(branch, cases, config["output_dir"])
        print(f"{branch}: selected {len(cases)} cases")

    if args.branch == "all":
        cross_cases = cross_model_cases()
        validate_cases("cross_model", cross_cases)
        write_cross_model(cross_cases)
        print(f"cross_model: selected {len(cross_cases)} cases")


if __name__ == "__main__":
    main()
