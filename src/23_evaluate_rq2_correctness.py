"""Evaluate output correctness for RQ2.

This script labels generated outputs for the three RQ2 task types with
objective correctness criteria: factual QA, math reasoning, and code generation.

    It is conservative by design: uncertain cases are labeled incorrect rather
    than requiring manual review, so the RQ2 pipeline can remain fully automatic.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import string
import subprocess
import tempfile
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import sympy as sp
except Exception:  # pragma: no cover - handled at runtime
    sp = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATIONS = ROOT / "outputs" / "rq1_formal_original_generations.csv"
DEFAULT_PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts.csv"
DEFAULT_OUTPUT = ROOT / "rq2" / "outputs" / "rq2_original_correctness_by_generation.csv"
DEFAULT_SUMMARY = ROOT / "rq2" / "outputs" / "rq2_original_correctness_summary_by_task.csv"

RQ2_TASKS = {"factual_qa", "math_reasoning", "code_generation"}


@dataclass
class EvaluationResult:
    extracted_answer: str
    is_correct: bool | None
    needs_manual_review: bool
    correctness_method: str
    notes: str
    performance_score: float | None = None
    factual_containment_match: bool | None = None
    factual_token_f1: float | None = None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str, remove_articles: bool = False) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\s+", " ", text).strip()
    if remove_articles:
        words = [word for word in text.split() if word not in {"a", "an", "the"}]
        text = " ".join(words)
    return text


def token_overlap_ratio(reference: str, output: str) -> float:
    ref_tokens = set(normalize_text(reference, remove_articles=True).split())
    out_tokens = set(normalize_text(output, remove_articles=True).split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & out_tokens) / len(ref_tokens)


def token_f1(reference: str, output: str) -> float:
    ref_tokens = normalize_text(reference, remove_articles=True).split()
    out_tokens = normalize_text(output, remove_articles=True).split()
    if not ref_tokens or not out_tokens:
        return 0.0

    common = Counter(ref_tokens) & Counter(out_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(out_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_factual_qa(output_text: str, reference_answer: str) -> EvaluationResult:
    normalized_ref = normalize_text(reference_answer, remove_articles=True)
    normalized_output = normalize_text(output_text, remove_articles=True)

    if not normalized_ref:
        return EvaluationResult(
            "",
            False,
            False,
            "factual_qa_containment_token_f1",
            "missing reference answer; labeled incorrect",
            performance_score=0.0,
            factual_containment_match=False,
            factual_token_f1=0.0,
        )

    if normalized_ref in normalized_output:
        return EvaluationResult(
            reference_answer,
            None,
            False,
            "factual_qa_containment_token_f1",
            "reference answer contained in output; factual_score=1.0; binary correctness not assigned for factual QA",
            performance_score=1.0,
            factual_containment_match=True,
            factual_token_f1=1.0,
        )

    overlap = token_overlap_ratio(reference_answer, output_text)
    f1 = token_f1(reference_answer, output_text)
    return EvaluationResult(
        reference_answer,
        None,
        False,
        "factual_qa_containment_token_f1",
        f"reference answer not contained; token overlap={overlap:.2f}; token_f1={f1:.6f}; binary correctness not assigned for factual QA",
        performance_score=f1,
        factual_containment_match=False,
        factual_token_f1=f1,
    )


ANSWER_PHRASE_RE = re.compile(
    r"(?:final answer|answer|therefore|thus)\s*(?:is|=|:)?\s*([^\n.]+)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?")


def extract_last_latex_command_argument(text: str, command: str) -> str:
    """Extract the final braced argument for a LaTeX command with nested braces."""
    marker = f"\\{command}"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        brace_start = text.find("{", start + len(marker))
        if brace_start == -1:
            continue
        depth = 0
        for index in range(brace_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1:index]
    return ""


def replace_latex_fractions(text: str) -> str:
    """Convert simple LaTeX fractions to a parser-friendly form."""
    text = re.sub(
        r"\\(?:dfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1)/(\2)",
        text,
    )
    return text


def clean_math_candidate(text: str) -> str:
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = replace_latex_fractions(text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace(",", "")
    text = text.strip(" .,:;")
    return text


def extract_math_answer(output_text: str) -> str:
    boxed = extract_last_latex_command_argument(output_text, "boxed")
    if boxed:
        return clean_math_candidate(boxed)

    phrase_matches = ANSWER_PHRASE_RE.findall(output_text)
    if phrase_matches:
        candidate = clean_math_candidate(phrase_matches[-1])
        number = NUMBER_RE.findall(candidate)
        if number:
            return clean_math_candidate(number[-1])
        if candidate:
            return candidate

    numbers = NUMBER_RE.findall(output_text)
    if numbers:
        return clean_math_candidate(numbers[-1])

    return ""


def normalize_math_string(text: str) -> str:
    text = clean_math_candidate(text).lower()
    text = re.sub(r"\s+", "", text)
    return text


def sympy_equivalent(left: str, right: str) -> bool | None:
    if sp is None:
        return None

    left = clean_math_candidate(left)
    right = clean_math_candidate(right)
    if not left or not right:
        return None

    try:
        left_expr = sp.sympify(left)
        right_expr = sp.sympify(right)
        diff = sp.simplify(left_expr - right_expr)
        return bool(diff == 0)
    except Exception:
        return None


def evaluate_math_reasoning(output_text: str, reference_answer: str) -> EvaluationResult:
    extracted = extract_math_answer(output_text)
    if not extracted:
        return EvaluationResult("", False, False, "math_final_answer_equivalence", "could not extract final answer; labeled incorrect", performance_score=0.0)

    normalized_extracted = normalize_math_string(extracted)
    normalized_reference = normalize_math_string(reference_answer)

    if normalized_extracted == normalized_reference:
        return EvaluationResult(extracted, True, False, "math_final_answer_equivalence", "normalized answer matched", performance_score=1.0)

    symbolic = sympy_equivalent(extracted, reference_answer)
    if symbolic is True:
        return EvaluationResult(extracted, True, False, "math_final_answer_equivalence", "symbolic equivalence matched", performance_score=1.0)
    if symbolic is False:
        return EvaluationResult(extracted, False, False, "math_final_answer_equivalence", "symbolic equivalence failed", performance_score=0.0)

    return EvaluationResult(
        extracted,
        False,
        False,
        "math_final_answer_equivalence",
        "could not safely compare extracted answer with reference; labeled incorrect",
        performance_score=0.0,
    )


def extract_python_code(output_text: str, entry_point: str | None = None, declaration: str | None = None) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", output_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()

    lines = output_text.splitlines()
    start_index = None
    if entry_point:
        pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
        for index, line in enumerate(lines):
            if pattern.search(line):
                start_index = index
                break

    if start_index is None:
        for index, line in enumerate(lines):
            if re.match(r"^\s*(from\s+\S+\s+import\s+|import\s+|def\s+\w+\s*\()", line):
                start_index = index
                break

    if start_index is not None:
        code = "\n".join(lines[start_index:]).strip()
    else:
        code = output_text.strip()

    if entry_point and f"def {entry_point}" not in code and declaration:
        maybe_body = "\n".join(f"    {line}" if line.strip() else line for line in code.splitlines())
        code = f"{declaration.rstrip()}\n{maybe_body}"

    return code.strip()


def load_humanevalpack_tests(source_indices: set[str]) -> dict[str, dict[str, str]]:
    try:
        from datasets import load_dataset
    except Exception:
        return {}

    try:
        dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
    except Exception:
        return {}

    tests: dict[str, dict[str, str]] = {}
    for source_index in source_indices:
        try:
            row = dataset[int(source_index)]
        except Exception:
            continue
        tests[source_index] = {
            "task_id": str(row.get("task_id", "")),
            "entry_point": str(row.get("entry_point", "")),
            "declaration": str(row.get("declaration", "")),
            "imports": str(row.get("import", "")),
            "test_setup": str(row.get("test_setup", "")),
            "test": str(row.get("test", "")),
        }
    return tests


def code_is_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def evaluate_code_generation(output_text: str, test_info: dict[str, str] | None) -> EvaluationResult:
    if not test_info:
        return EvaluationResult("", None, True, "humanevalpack_functional_tests", "missing HumanEvalPack test metadata", performance_score=None)

    entry_point = test_info.get("entry_point", "")
    code = extract_python_code(output_text, entry_point=entry_point, declaration=test_info.get("declaration", ""))

    if not code or not entry_point:
        return EvaluationResult(code, None, True, "humanevalpack_functional_tests", "missing extracted code or entry point", performance_score=None)

    if not code_is_parseable(code):
        return EvaluationResult(code, False, False, "humanevalpack_functional_tests", "extracted code has syntax error", performance_score=0.0)

    runner = "\n\n".join(
        part
        for part in [
            test_info.get("imports", ""),
            test_info.get("test_setup", ""),
            code,
            test_info.get("test", ""),
        ]
        if part.strip()
    )

    with tempfile.TemporaryDirectory(prefix="rq2_code_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner, encoding="utf-8")
        try:
            completed = subprocess.run(
                ["python3", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return EvaluationResult(code, False, False, "humanevalpack_functional_tests", "execution timed out", performance_score=0.0)

    if completed.returncode == 0:
        return EvaluationResult(code, True, False, "humanevalpack_functional_tests", "passed HumanEvalPack tests", performance_score=1.0)

    error_tail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
    note = error_tail[0] if error_tail else f"test process returned {completed.returncode}"
    return EvaluationResult(code, False, False, "humanevalpack_functional_tests", note[:300], performance_score=0.0)


def bool_to_csv(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def evaluate_row(row: dict[str, str], prompt_meta: dict[str, str], code_tests: dict[str, dict[str, str]]) -> EvaluationResult:
    task_type = row["task_type"]
    output_text = row.get("output_text", "")
    reference_answer = prompt_meta.get("reference_answer", "")

    if task_type == "factual_qa":
        return evaluate_factual_qa(output_text, reference_answer)
    if task_type == "math_reasoning":
        return evaluate_math_reasoning(output_text, reference_answer)
    if task_type == "code_generation":
        return evaluate_code_generation(output_text, code_tests.get(row.get("source_index", "")))

    return EvaluationResult("", None, True, "unsupported_task_type", f"task excluded from RQ2: {task_type}", performance_score=None)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)

    summary_rows = []
    for task_type, task_rows in sorted(grouped.items()):
        auto_rows = [row for row in task_rows if row["needs_manual_review"] != "true"]
        n_correct = sum(row["is_correct"] == "true" for row in task_rows)
        n_incorrect = sum(row["is_correct"] == "false" for row in task_rows)
        n_unlabeled = sum(row["is_correct"] == "" for row in task_rows)
        n_manual = sum(row["needs_manual_review"] == "true" for row in task_rows)
        rate = ""
        mean_score = ""
        if auto_rows:
            labeled_rows = [row for row in auto_rows if row["is_correct"] in {"true", "false"}]
            if labeled_rows:
                rate = f"{sum(row['is_correct'] == 'true' for row in labeled_rows) / len(labeled_rows):.6f}"
            scores = [
                float(row["performance_score"])
                for row in auto_rows
                if row.get("performance_score", "") != ""
            ]
            if scores:
                mean_score = f"{sum(scores) / len(scores):.6f}"
        summary_rows.append(
            {
                "task_type": task_type,
                "n_outputs": str(len(task_rows)),
                "n_correct": str(n_correct),
                "n_incorrect": str(n_incorrect),
                "n_unlabeled_correctness": str(n_unlabeled),
                "n_manual_review": str(n_manual),
                "correct_rate_auto_only": rate,
                "mean_performance_score": mean_score,
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RQ2 correctness labels.")
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompt-variant", default="")
    parser.add_argument("--perturbation-type", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_rows = read_csv_rows(args.prompts)
    prompt_by_item = {row["item_id"]: row for row in prompt_rows}

    generation_rows = [
        row for row in read_csv_rows(args.generations)
        if row.get("task_type") in RQ2_TASKS
    ]

    code_source_indices = {
        row.get("source_index", "")
        for row in generation_rows
        if row.get("task_type") == "code_generation"
    }
    code_tests = load_humanevalpack_tests(code_source_indices)

    output_rows: list[dict[str, str]] = []
    for row in generation_rows:
        prompt_meta = prompt_by_item.get(row["item_id"], {})
        result = evaluate_row(row, prompt_meta, code_tests)
        prompt_variant = args.prompt_variant
        if not prompt_variant:
            prompt_variant = "perturbed" if row.get("perturbed_prompt") else "original"
        output_rows.append(
            {
                "item_id": row.get("item_id", ""),
                "task_type": row.get("task_type", ""),
                "dataset_name": row.get("dataset_name", ""),
                "source_index": row.get("source_index", ""),
                "source_id": prompt_meta.get("source_id", ""),
                "sample_id": row.get("sample_id", ""),
                "model_name": row.get("model_name", ""),
                "prompt_variant": prompt_variant,
                "perturbation_type": row.get("perturbation_type", args.perturbation_type),
                "reference_answer": prompt_meta.get("reference_answer", ""),
                "output_text": row.get("output_text", ""),
                "extracted_answer": result.extracted_answer,
                "performance_score": "" if result.performance_score is None else f"{result.performance_score:.6f}",
                "factual_containment_match": "" if result.factual_containment_match is None else ("true" if result.factual_containment_match else "false"),
                "factual_token_f1": "" if result.factual_token_f1 is None else f"{result.factual_token_f1:.6f}",
                "is_correct": bool_to_csv(result.is_correct),
                "needs_manual_review": "true" if result.needs_manual_review else "false",
                "correctness_method": result.correctness_method,
                "notes": result.notes,
            }
        )

    fieldnames = [
        "item_id",
        "task_type",
        "dataset_name",
        "source_index",
        "source_id",
        "sample_id",
        "model_name",
        "prompt_variant",
        "perturbation_type",
        "reference_answer",
        "output_text",
        "extracted_answer",
        "performance_score",
        "factual_containment_match",
        "factual_token_f1",
        "is_correct",
        "needs_manual_review",
        "correctness_method",
        "notes",
    ]
    write_csv_rows(args.output, output_rows, fieldnames)

    summary_rows = summarize(output_rows)
    write_csv_rows(
        args.summary,
        summary_rows,
        [
            "task_type",
            "n_outputs",
            "n_correct",
            "n_incorrect",
            "n_unlabeled_correctness",
            "n_manual_review",
            "correct_rate_auto_only",
            "mean_performance_score",
        ],
    )

    print(f"Wrote {len(output_rows)} correctness rows to {args.output}")
    print(f"Wrote summary to {args.summary}")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
