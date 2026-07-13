"""Repair math reasoning paraphrase prompts.

The original math paraphrase prompts contain template artifacts such as
Research Question, Rewrite, Code Signature, Task Signature, fenced code blocks,
and graph prompts where the original [asy] diagram was removed. This script
creates fixed prompt files without overwriting the original prompt CSV.

Outputs:
    prompts/rq1_formal_perturbed_prompts_n50_math_reasoning_paraphrasing_fixed.csv
    prompts/rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv
    outputs/math_paraphrase_prompt_repair_report.csv
    outputs/math_paraphrase_prompt_repair_report.md
"""

from __future__ import annotations

import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning.csv"
PARAPHRASE_FIXED = (
    ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning_paraphrasing_fixed.csv"
)
FULL_FIXED = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv"
REPORT_CSV = ROOT / "outputs" / "math_paraphrase_prompt_repair_report.csv"
REPORT_MD = ROOT / "outputs" / "math_paraphrase_prompt_repair_report.md"

ARTIFACT_RE = re.compile(
    r"Research Question\s*:|Rewrite(?:\s+the\s+prompt[^:\n]*)?\s*:|"
    r"Code Signature\s*:|Code signature\s*:|Task Signature\s*:|```|"
    r"def\s+\w+\s*\(|function\s+\w+\s*\(",
    flags=re.IGNORECASE,
)


def normalize_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_original_problem(prompt: str) -> str:
    text = normalize_space(prompt)
    text = re.sub(r"(?is)^\s*Problem\s*:\s*", "", text)
    text = re.split(r"(?im)^\s*Instruction\s*:\s*", text, maxsplit=1)[0]
    return normalize_space(text)


def remove_fenced_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def remove_signature_sections(text: str) -> str:
    text = remove_fenced_blocks(text)
    text = re.sub(
        r"(?is)\n?\s*(?:\*\*)?(?:Code Signature|Code signature|Task Signature|Code)(?:\*\*)?\s*:.*$",
        "",
        text,
    )
    return text


def strip_rewrite_preamble(text: str) -> str:
    patterns = [
        r"(?is)^\s*Rewrite\s+the\s+prompt\s+while\s+(?:maintaining|preserving|keeping).*?:\s*",
        r"(?is)^\s*Rewrite\s+the\s+prompt\s+without\s+changing\s+its\s+meaning\s*:\s*",
        r"(?is)^\s*Rewrite\s+the\s+prompt\s*:\s*",
        r"(?is)^\s*Rewrite\s*:\s*",
    ]
    changed = True
    text = text.strip()
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", text, count=1)
            if updated != text:
                text = updated.strip()
                changed = True
                break
    return text


def remove_trailing_task_sections(text: str) -> str:
    text = re.sub(
        r"(?ims)\n\s*(?:Task|Action)\s*:\s*(?:Calculate|Compute|Solve|Find|Determine|Report|Provide|Present|Use|You should|Return)\b.*$",
        "",
        text,
    )
    text = re.sub(
        r"(?ims)\n\s*You should solve the problem.*$",
        "",
        text,
    )
    text = re.sub(
        r"(?ims)\n\s*Instructions?\s*:\s*Solve the problem.*$",
        "",
        text,
    )
    return text


def clean_paraphrased_problem(prompt: str) -> str:
    text = normalize_space(prompt)
    text = remove_signature_sections(text)
    text = strip_rewrite_preamble(text)
    text = re.sub(r"(?im)^\s*(?:Research Question|Task|Action|Problem)\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*-\s*(?:Input|Output)\s*:.*$", "", text)
    text = re.split(r"(?im)^\s*Instruction\s*:\s*", text, maxsplit=1)[0]
    text = remove_trailing_task_sections(text)
    text = strip_rewrite_preamble(text)
    text = normalize_space(text)
    return text


def asy_blocks(text: str) -> list[str]:
    return [match.group(0).strip() for match in re.finditer(r"\[asy\].*?\[/asy\]", text, flags=re.DOTALL)]


def repair_prompt(original_prompt: str, perturbed_prompt: str) -> tuple[str, dict[str, object]]:
    original_problem = extract_original_problem(original_prompt)
    clean_problem = clean_paraphrased_problem(perturbed_prompt)
    if not clean_problem:
        clean_problem = original_problem

    original_asy = asy_blocks(original_problem)
    clean_has_asy = "[asy]" in clean_problem
    asy_restored = bool(original_asy and not clean_has_asy)
    if asy_restored:
        clean_problem = normalize_space(clean_problem)
        clean_problem = f"{clean_problem}\n\nDiagram:\n" + "\n\n".join(original_asy)

    fixed_prompt = (
        "Problem: "
        + normalize_space(clean_problem)
        + "\n\nInstruction: Solve the problem and provide the final answer."
    )

    before_artifact = bool(ARTIFACT_RE.search(str(perturbed_prompt)))
    after_artifact = bool(ARTIFACT_RE.search(fixed_prompt))
    before_asy_removed = bool(original_asy and "[asy]" not in str(perturbed_prompt))
    after_asy_removed = bool(original_asy and "[asy]" not in fixed_prompt)
    report = {
        "had_template_artifact": before_artifact,
        "fixed_has_template_artifact": after_artifact,
        "had_asy_removed": before_asy_removed,
        "fixed_has_asy_removed": after_asy_removed,
        "asy_restored": asy_restored,
        "original_prompt_length_chars": len(str(original_prompt)),
        "old_perturbed_prompt_length_chars": len(str(perturbed_prompt)),
        "fixed_perturbed_prompt_length_chars": len(fixed_prompt),
    }
    return fixed_prompt, report


def validate(paraphrase_df: pd.DataFrame, report_df: pd.DataFrame, expected_rows: int) -> None:
    failures = []
    if len(paraphrase_df) != expected_rows:
        failures.append(f"expected {expected_rows} paraphrase rows, found {len(paraphrase_df)}")
    if paraphrase_df["item_id"].nunique() != expected_rows:
        failures.append(f"expected {expected_rows} unique paraphrase item_id values")
    if not (paraphrase_df["perturbation_type"] == "paraphrasing").all():
        failures.append("paraphrase fixed file contains non-paraphrasing rows")
    if report_df["fixed_has_template_artifact"].any():
        bad = report_df.loc[report_df["fixed_has_template_artifact"], "item_id"].tolist()
        failures.append(f"fixed prompts still have template artifacts: {bad[:10]}")
    if report_df["fixed_has_asy_removed"].any():
        bad = report_df.loc[report_df["fixed_has_asy_removed"], "item_id"].tolist()
        failures.append(f"fixed prompts still remove ASY graph blocks: {bad[:10]}")
    if paraphrase_df["perturbed_prompt"].isna().any():
        failures.append("fixed perturbed_prompt contains NA")
    if failures:
        raise SystemExit("validation failed: " + "; ".join(failures))


def write_report(report_df: pd.DataFrame, report_md: Path) -> None:
    summary = {
        "n_paraphrase_rows": len(report_df),
        "template_artifact_before": int(report_df["had_template_artifact"].sum()),
        "template_artifact_after": int(report_df["fixed_has_template_artifact"].sum()),
        "asy_removed_before": int(report_df["had_asy_removed"].sum()),
        "asy_removed_after": int(report_df["fixed_has_asy_removed"].sum()),
        "asy_restored": int(report_df["asy_restored"].sum()),
    }
    lines = [
        "# Math Paraphrase Prompt Repair Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "Affected rows before repair:",
            "",
            report_df[
                report_df["had_template_artifact"] | report_df["had_asy_removed"]
            ][
                [
                    "item_id",
                    "had_template_artifact",
                    "had_asy_removed",
                    "asy_restored",
                    "fixed_has_template_artifact",
                    "fixed_has_asy_removed",
                ]
            ].to_markdown(index=False),
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--paraphrase-output", type=Path, default=PARAPHRASE_FIXED)
    parser.add_argument("--full-output", type=Path, default=FULL_FIXED)
    parser.add_argument("--report-csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--expected-paraphrase-rows", type=int, default=50)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    paraphrase_output = (
        args.paraphrase_output
        if args.paraphrase_output.is_absolute()
        else ROOT / args.paraphrase_output
    )
    full_output = args.full_output if args.full_output.is_absolute() else ROOT / args.full_output
    report_csv = args.report_csv if args.report_csv.is_absolute() else ROOT / args.report_csv
    report_md = args.report_md if args.report_md.is_absolute() else ROOT / args.report_md

    df = pd.read_csv(input_path)
    full_fixed = df.copy()
    reports = []

    mask = full_fixed["perturbation_type"] == "paraphrasing"
    for index, row in full_fixed[mask].iterrows():
        fixed_prompt, report = repair_prompt(row["original_prompt"], row["perturbed_prompt"])
        full_fixed.at[index, "perturbed_prompt"] = fixed_prompt
        reports.append({"item_id": row["item_id"], **report})

    paraphrase_fixed = full_fixed[mask].copy()
    report_df = pd.DataFrame(reports)
    validate(paraphrase_fixed, report_df, args.expected_paraphrase_rows)

    paraphrase_output.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    paraphrase_fixed.to_csv(paraphrase_output, index=False)
    full_fixed.to_csv(full_output, index=False)
    report_df.to_csv(report_csv, index=False)
    write_report(report_df, report_md)

    print(f"Wrote {paraphrase_output}")
    print(f"Wrote {full_output}")
    print(f"Wrote {report_csv}")
    print(f"Wrote {report_md}")
    print(report_df[["had_template_artifact", "fixed_has_template_artifact", "had_asy_removed", "fixed_has_asy_removed"]].sum())


if __name__ == "__main__":
    main()
