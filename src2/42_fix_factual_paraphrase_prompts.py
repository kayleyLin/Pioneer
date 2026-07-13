"""Repair factual-QA paraphrasing prompts so they retain the original context.

The existing factual-QA paraphrasing CSV contains many paraphrasing rows whose
`perturbed_prompt` is only the rewritten question. Some rows also include a
rewrite-instruction prefix. This script extracts the actual paraphrased question
and reconstructs a full factual-QA prompt:

    Context: <original context>

    Question: <paraphrased question>

It writes a new CSV and does not overwrite the original prompt file.

Outputs:
    prompts/rq1_formal_perturbed_prompts_n50_factual_qa_fixed.csv
    outputs/factual_paraphrase_prompt_repair_report.csv
    outputs/factual_paraphrase_prompt_repair_report.md
"""

from __future__ import annotations

import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_factual_qa.csv"
OUTPUT = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_factual_qa_fixed.csv"
REPORT_CSV = ROOT / "outputs" / "factual_paraphrase_prompt_repair_report.csv"
REPORT_MD = ROOT / "outputs" / "factual_paraphrase_prompt_repair_report.md"

TASK = "factual_qa"
PERTURBATION = "paraphrasing"


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


def strip_rewrite_prefix(text: str) -> str:
    text = text.strip()
    patterns = [
        r"(?is)^\s*Rewrite\s+the\s+prompt\s+while\s+(?:maintaining|preserving|keeping).*?:\s*",
        r"(?is)^\s*Rewrite\s+the\s+prompt\s+without\s+changing\s+its\s+meaning\s*:\s*",
        r"(?is)^\s*Rewrite\s+the\s+prompt\s*:\s*",
        r"(?is)^\s*Rewrite\s*:\s*",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", text, count=1)
            if updated != text:
                text = updated.strip()
                changed = True
                break
    return text


def clean_paraphrased_question(parsed: dict[str, str]) -> str:
    question = strip_rewrite_prefix(parsed["question"])
    question_marker = list(re.finditer(r"(?:^|\n)\s*Question:\s*", question, flags=re.IGNORECASE))
    if question_marker:
        question = question[question_marker[-1].end() :].strip()
    return strip_rewrite_prefix(question)


def validate_fixed(df: pd.DataFrame, expected_rows: int) -> None:
    paraphrase = df[(df["task_type"] == TASK) & (df["perturbation_type"] == PERTURBATION)].copy()
    if len(paraphrase) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} factual-QA paraphrasing rows, found {len(paraphrase)}")

    statuses = paraphrase["perturbed_prompt"].apply(parse_context_question).apply(lambda parsed: parsed["status"])
    bad_status = statuses.ne("parsed_full_context_question")
    if bad_status.any():
        bad_ids = paraphrase.loc[bad_status, "item_id"].head(10).tolist()
        raise SystemExit(f"Fixed prompts still have malformed paraphrasing rows: {bad_ids}")

    residue = paraphrase["perturbed_prompt"].apply(has_rewrite_prefix)
    if residue.any():
        bad_ids = paraphrase.loc[residue, "item_id"].head(10).tolist()
        raise SystemExit(f"Fixed prompts still contain rewrite-prefix residue: {bad_ids}")


def build_report(original: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, original_row in original.iterrows():
        fixed_row = fixed.loc[original_row.name]
        if original_row["task_type"] != TASK or original_row["perturbation_type"] != PERTURBATION:
            continue

        original_parse = parse_context_question(original_row["original_prompt"])
        perturbed_before = parse_context_question(original_row["perturbed_prompt"])
        perturbed_after = parse_context_question(fixed_row["perturbed_prompt"])
        rows.append(
            {
                "item_id": original_row["item_id"],
                "before_status": perturbed_before["status"],
                "after_status": perturbed_after["status"],
                "before_had_rewrite_prefix": bool(perturbed_before["prefix"]) or has_rewrite_prefix(original_row["perturbed_prompt"]),
                "after_has_rewrite_prefix": has_rewrite_prefix(fixed_row["perturbed_prompt"]),
                "original_context_chars": len(original_parse["context"]),
                "before_context_chars": len(perturbed_before["context"]),
                "after_context_chars": len(perturbed_after["context"]),
                "paraphrased_question": perturbed_after["question"],
                "fixed_prompt": fixed_row["perturbed_prompt"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown(report: pd.DataFrame, report_md: Path) -> None:
    before_counts = report["before_status"].value_counts().rename_axis("status").reset_index(name="rows")
    after_counts = report["after_status"].value_counts().rename_axis("status").reset_index(name="rows")
    before_prefix = int(report["before_had_rewrite_prefix"].sum())
    after_prefix = int(report["after_has_rewrite_prefix"].sum())
    lines = [
        "# Factual QA Paraphrasing Prompt Repair Report",
        "",
        f"- Repaired rows: {len(report)}",
        f"- Rows with rewrite-prefix residue before repair: {before_prefix}",
        f"- Rows with rewrite-prefix residue after repair: {after_prefix}",
        "",
        "## Before Parse Status",
        "",
        before_counts.to_markdown(index=False),
        "",
        "## After Parse Status",
        "",
        after_counts.to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        "PASS. The fixed factual-QA paraphrasing prompts all contain a full `Context:` block and a clean `Question:` block.",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report-csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--expected-paraphrase-rows", type=int, default=50)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report_csv = args.report_csv if args.report_csv.is_absolute() else ROOT / args.report_csv
    report_md = args.report_md if args.report_md.is_absolute() else ROOT / args.report_md

    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    fixed = df.copy()
    report_rows = []

    for index, row in df.iterrows():
        if row["task_type"] != TASK or row["perturbation_type"] != PERTURBATION:
            continue

        original_parse = parse_context_question(row["original_prompt"])
        perturbed_parse = parse_context_question(row["perturbed_prompt"])
        if not original_parse["context"] or not original_parse["question"]:
            raise SystemExit(f"Could not parse original prompt for {row['item_id']}")
        paraphrased_question = clean_paraphrased_question(perturbed_parse)
        if not paraphrased_question:
            raise SystemExit(f"Could not parse paraphrased question for {row['item_id']}")

        repaired_prompt = (
            f"Context: {original_parse['context']}\n\n"
            f"Question: {paraphrased_question}"
        )
        fixed.loc[index, "perturbed_prompt"] = repaired_prompt
        fixed.loc[index, "construction_method"] = "llm_assisted_question_paraphrase_context_preserved_repair"
        fixed.loc[index, "method_reference"] = (
            "Post-hoc repair: original context restored around paraphrased question"
        )
        report_rows.append(index)

    validate_fixed(fixed, args.expected_paraphrase_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_path, index=False)

    report = build_report(df, fixed)
    report.to_csv(report_csv, index=False)
    write_markdown(report, report_md)

    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(f"Wrote {report_csv.relative_to(ROOT)}")
    print(f"Wrote {report_md.relative_to(ROOT)}")
    print(f"Repaired factual-QA paraphrasing rows: {len(report_rows)}")


if __name__ == "__main__":
    main()
