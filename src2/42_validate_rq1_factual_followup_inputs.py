"""Validate inputs for the factual-QA paraphrasing follow-up.

This script does not call any model API. It checks that the ChatGPT/main branch
and the Qwen branch have complete RQ1 n=50 original, perturbed, and SBERT
analysis files before the explanatory follow-up scripts are run.

Outputs:
    outputs/factual_followup_input_validation.csv
    outputs/factual_followup_input_validation.md
    qwen/outputs/factual_followup_input_validation.csv
    qwen/outputs/factual_followup_input_validation.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TASKS = [
    "factual_qa",
    "math_reasoning",
    "code_generation",
    "open_ended_writing",
]

PERTURBATIONS = [
    "context_injection",
    "formatting_changes",
    "paraphrasing",
    "reordering",
    "surface_noise",
]

BRANCHES = {
    "outputs": {
        "output_dir": ROOT / "outputs",
        "original_pattern": "rq1_formal_original_generations_n50_{task}.csv",
        "perturbed_pattern": "rq1_formal_perturbed_generations_n50_{task}.csv",
    },
    "qwen": {
        "output_dir": ROOT / "qwen" / "outputs",
        "original_pattern": "rq1_qwen_original_generations_n50_{task}.csv",
        "perturbed_pattern": "rq1_qwen_perturbed_generations_n50_{task}.csv",
    },
}

ORIGINAL_PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50_factual_qa.csv"
DEFAULT_PERTURBED_PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_factual_qa.csv"


def status_from_counts(actual: int, expected: int) -> str:
    return "PASS" if actual == expected else "FAIL"


def clean_space(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_context_question(prompt: object) -> dict[str, str]:
    raw = "" if pd.isna(prompt) else str(prompt).strip()
    if not raw:
        return {"context": "", "question": "", "status": "empty", "prefix": ""}

    context_match = re.search(r"(?:^|\n)\s*Context:\s*", raw, flags=re.IGNORECASE)
    question_matches = list(re.finditer(r"(?:^|\n)\s*Question:\s*", raw, flags=re.IGNORECASE))
    if context_match:
        question_matches = [match for match in question_matches if match.start() > context_match.end()]
    if context_match and question_matches:
        question_match = question_matches[-1]
        context = raw[context_match.end() : question_match.start()].strip()
        question = raw[question_match.end() :].strip()
        prefix = raw[: context_match.start()].strip()
        if context and question and prefix:
            status = "parsed_full_context_question_with_prefix"
        elif context and question:
            status = "parsed_full_context_question"
        else:
            status = "partial_context_question"
        return {"context": context, "question": question, "status": status, "prefix": prefix}

    question_only = re.match(r"^\s*Question:\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if question_only:
        return {
            "context": "",
            "question": question_only.group(1).strip(),
            "status": "question_only_with_marker",
            "prefix": "",
        }

    return {"context": "", "question": raw, "status": "question_only_no_marker", "prefix": ""}


def has_rewrite_prefix(text: object) -> bool:
    raw = "" if pd.isna(text) else str(text)
    return bool(
        re.search(
            r"\b(rewrite|paraphrase)\b.*\b(prompt|question)\b",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def add_row(
    rows: list[dict[str, object]],
    branch: str,
    check: str,
    expected: object,
    actual: object,
    status: str,
    details: str = "",
) -> None:
    rows.append(
        {
            "branch": branch,
            "check": check,
            "expected": expected,
            "actual": actual,
            "status": status,
            "details": details,
        }
    )


def read_required_csv(
    path: Path,
    rows: list[dict[str, object]],
    branch: str,
    label: str,
) -> pd.DataFrame | None:
    if not path.exists():
        add_row(rows, branch, f"file_exists:{label}", "exists", "missing", "FAIL", str(path))
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive validation path
        add_row(rows, branch, f"file_readable:{label}", "readable", type(exc).__name__, "FAIL", str(path))
        return None
    add_row(rows, branch, f"file_exists:{label}", "exists", "exists", "PASS", str(path))
    return df


def validate_original_generations(branch: str, config: dict[str, object], rows: list[dict[str, object]]) -> None:
    output_dir = config["output_dir"]
    pattern = config["original_pattern"]
    assert isinstance(output_dir, Path)
    assert isinstance(pattern, str)

    all_parts = []
    for task in TASKS:
        path = output_dir / pattern.format(task=task)
        df = read_required_csv(path, rows, branch, f"original_generations:{task}")
        if df is None:
            continue

        task_df = df[df["task_type"] == task].copy()
        item_count = task_df["item_id"].nunique()
        add_row(
            rows,
            branch,
            f"original_item_count:{task}",
            50,
            item_count,
            status_from_counts(item_count, 50),
        )

        bad_cells = (
            task_df.groupby(["task_type", "item_id"], dropna=False)["sample_id"]
            .nunique()
            .reset_index(name="n_samples")
        )
        bad_cells = bad_cells[bad_cells["n_samples"] != 5]
        add_row(
            rows,
            branch,
            f"original_sample_cells:{task}",
            "50 cells with 5 samples",
            len(bad_cells),
            "PASS" if bad_cells.empty and item_count == 50 else "FAIL",
            "actual reports bad cell count",
        )

        duplicate_samples = task_df.duplicated(["task_type", "item_id", "sample_id"]).sum()
        add_row(
            rows,
            branch,
            f"original_duplicate_sample_rows:{task}",
            0,
            int(duplicate_samples),
            status_from_counts(int(duplicate_samples), 0),
        )

        empty_outputs = task_df["output_text"].fillna("").astype(str).str.strip().eq("").sum()
        add_row(
            rows,
            branch,
            f"original_empty_outputs:{task}",
            0,
            int(empty_outputs),
            status_from_counts(int(empty_outputs), 0),
        )
        all_parts.append(task_df)

    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        total_rows = len(combined)
        add_row(rows, branch, "original_generation_total_rows", 1000, total_rows, status_from_counts(total_rows, 1000))


def validate_perturbed_generations(branch: str, config: dict[str, object], rows: list[dict[str, object]]) -> None:
    output_dir = config["output_dir"]
    pattern = config["perturbed_pattern"]
    assert isinstance(output_dir, Path)
    assert isinstance(pattern, str)

    all_parts = []
    for task in TASKS:
        path = output_dir / pattern.format(task=task)
        df = read_required_csv(path, rows, branch, f"perturbed_generations:{task}")
        if df is None:
            continue

        task_df = df[df["task_type"] == task].copy()
        for perturbation in PERTURBATIONS:
            cell_df = task_df[task_df["perturbation_type"] == perturbation].copy()
            item_count = cell_df["item_id"].nunique()
            add_row(
                rows,
                branch,
                f"perturbed_item_count:{task}:{perturbation}",
                50,
                item_count,
                status_from_counts(item_count, 50),
            )

            bad_cells = (
                cell_df.groupby(["task_type", "perturbation_type", "item_id"], dropna=False)["sample_id"]
                .nunique()
                .reset_index(name="n_samples")
            )
            bad_cells = bad_cells[bad_cells["n_samples"] != 5]
            add_row(
                rows,
                branch,
                f"perturbed_sample_cells:{task}:{perturbation}",
                "50 cells with 5 samples",
                len(bad_cells),
                "PASS" if bad_cells.empty and item_count == 50 else "FAIL",
                "actual reports bad cell count",
            )

            duplicate_samples = cell_df.duplicated(
                ["task_type", "perturbation_type", "item_id", "sample_id"]
            ).sum()
            add_row(
                rows,
                branch,
                f"perturbed_duplicate_sample_rows:{task}:{perturbation}",
                0,
                int(duplicate_samples),
                status_from_counts(int(duplicate_samples), 0),
            )

            empty_outputs = cell_df["output_text"].fillna("").astype(str).str.strip().eq("").sum()
            add_row(
                rows,
                branch,
                f"perturbed_empty_outputs:{task}:{perturbation}",
                0,
                int(empty_outputs),
                status_from_counts(int(empty_outputs), 0),
            )
        all_parts.append(task_df)

    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        total_rows = len(combined)
        add_row(rows, branch, "perturbed_generation_total_rows", 5000, total_rows, status_from_counts(total_rows, 5000))


def validate_sbert_outputs(branch: str, output_dir: Path, rows: list[dict[str, object]]) -> None:
    effects = read_required_csv(
        output_dir / "sbert_rq1_n50_perturbation_effects_by_item.csv",
        rows,
        branch,
        "sbert_effects_by_item",
    )
    if effects is not None:
        row_count = len(effects)
        add_row(rows, branch, "sbert_effects_total_rows", 1000, row_count, status_from_counts(row_count, 1000))
        for task in TASKS:
            for perturbation in PERTURBATIONS:
                cell_count = len(
                    effects[
                        (effects["task_type"] == task)
                        & (effects["perturbation_type"] == perturbation)
                    ]
                )
                add_row(
                    rows,
                    branch,
                    f"sbert_effects_cell_rows:{task}:{perturbation}",
                    50,
                    cell_count,
                    status_from_counts(cell_count, 50),
                )

    summary = read_required_csv(
        output_dir / "sbert_rq1_n50_perturbation_summary.csv",
        rows,
        branch,
        "sbert_perturbation_summary",
    )
    if summary is not None:
        row_count = len(summary)
        add_row(rows, branch, "sbert_summary_total_rows", 20, row_count, status_from_counts(row_count, 20))


def validate_prompt_files(branch: str, rows: list[dict[str, object]], perturbed_prompts_path: Path) -> None:
    original = read_required_csv(ORIGINAL_PROMPTS, rows, branch, "original_factual_qa_prompts")
    if original is not None:
        factual = original[original["task_type"] == "factual_qa"].copy()
        item_count = factual["item_id"].nunique()
        add_row(rows, branch, "factual_original_prompt_items", 50, item_count, status_from_counts(item_count, 50))

        missing_refs = factual["reference_answer"].fillna("").astype(str).str.strip().eq("").sum()
        add_row(
            rows,
            branch,
            "factual_reference_answer_missing",
            0,
            int(missing_refs),
            status_from_counts(int(missing_refs), 0),
        )

        duplicate_items = factual.duplicated(["item_id"]).sum()
        add_row(
            rows,
            branch,
            "factual_original_prompt_duplicate_items",
            0,
            int(duplicate_items),
            status_from_counts(int(duplicate_items), 0),
        )

    perturbed = read_required_csv(perturbed_prompts_path, rows, branch, "perturbed_factual_qa_prompts")
    if perturbed is not None:
        paraphrase = perturbed[
            (perturbed["task_type"] == "factual_qa")
            & (perturbed["perturbation_type"] == "paraphrasing")
        ].copy()
        row_count = len(paraphrase)
        item_count = paraphrase["item_id"].nunique()
        add_row(rows, branch, "factual_paraphrase_prompt_rows", 50, row_count, status_from_counts(row_count, 50))
        add_row(rows, branch, "factual_paraphrase_prompt_items", 50, item_count, status_from_counts(item_count, 50))

        missing_prompts = (
            paraphrase["original_prompt"].fillna("").astype(str).str.strip().eq("")
            | paraphrase["perturbed_prompt"].fillna("").astype(str).str.strip().eq("")
        ).sum()
        add_row(
            rows,
            branch,
            "factual_paraphrase_prompt_missing_text",
            0,
            int(missing_prompts),
            status_from_counts(int(missing_prompts), 0),
        )

        parsed_original = paraphrase["original_prompt"].apply(parse_context_question)
        parsed_perturbed = paraphrase["perturbed_prompt"].apply(parse_context_question)
        malformed = parsed_perturbed.apply(lambda value: value["status"] != "parsed_full_context_question")
        add_row(
            rows,
            branch,
            "factual_paraphrase_prompt_full_context_question_rows",
            50,
            int((~malformed).sum()),
            status_from_counts(int((~malformed).sum()), 50),
            "paraphrasing perturbed_prompt must contain full Context and Question without a prefix",
        )

        context_mismatch = []
        for original_parse, perturbed_parse in zip(parsed_original, parsed_perturbed):
            context_mismatch.append(clean_space(original_parse["context"]) != clean_space(perturbed_parse["context"]))
        add_row(
            rows,
            branch,
            "factual_paraphrase_context_mismatch_rows",
            0,
            int(sum(context_mismatch)),
            status_from_counts(int(sum(context_mismatch)), 0),
        )

        rewrite_prefix_rows = paraphrase["perturbed_prompt"].apply(has_rewrite_prefix).sum()
        add_row(
            rows,
            branch,
            "factual_paraphrase_rewrite_prefix_rows",
            0,
            int(rewrite_prefix_rows),
            status_from_counts(int(rewrite_prefix_rows), 0),
        )


def write_outputs(branch: str, rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "factual_followup_input_validation.csv"
    md_path = output_dir / "factual_followup_input_validation.md"
    df.to_csv(csv_path, index=False)

    n_fail = int((df["status"] != "PASS").sum())
    lines = [
        f"# Factual Follow-up Input Validation: {branch}",
        "",
        f"- Total checks: {len(df)}",
        f"- Passing checks: {len(df) - n_fail}",
        f"- Failing checks: {n_fail}",
        "",
    ]
    if n_fail:
        lines.extend(["## Failing Checks", ""])
        failed = df[df["status"] != "PASS"]
        lines.append(failed.to_markdown(index=False))
        lines.append("")
    else:
        lines.extend(
            [
                "## Conclusion",
                "",
                "PASS. No missing item, sample, perturbation, SBERT, prompt, or reference-answer cells were detected.",
                "",
            ]
        )

    lines.extend(["## Full Check Table", "", df.to_markdown(index=False), ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


def validate_branch(branch: str, perturbed_prompts_path: Path) -> bool:
    config = BRANCHES[branch]
    output_dir = config["output_dir"]
    assert isinstance(output_dir, Path)

    rows: list[dict[str, object]] = []
    validate_original_generations(branch, config, rows)
    validate_perturbed_generations(branch, config, rows)
    validate_sbert_outputs(branch, output_dir, rows)
    validate_prompt_files(branch, rows, perturbed_prompts_path)
    write_outputs(branch, rows, output_dir)
    return all(row["status"] == "PASS" for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=["outputs", "qwen", "all"],
        default="all",
        help="Branch to validate. Defaults to all primary branches.",
    )
    parser.add_argument(
        "--perturbed-prompts",
        type=Path,
        default=DEFAULT_PERTURBED_PROMPTS,
        help="Factual-QA perturbed prompt CSV to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branches = ["outputs", "qwen"] if args.branch == "all" else [args.branch]
    perturbed_prompts_path = args.perturbed_prompts
    if not perturbed_prompts_path.is_absolute():
        perturbed_prompts_path = ROOT / perturbed_prompts_path
    results = {branch: validate_branch(branch, perturbed_prompts_path) for branch in branches}
    failed = [branch for branch, passed in results.items() if not passed]
    if failed:
        raise SystemExit(f"Validation failed for: {', '.join(failed)}")
    print("All requested branches passed factual follow-up input validation.")


if __name__ == "__main__":
    main()
