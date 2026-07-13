"""Compare deterministic math solution trajectories under original/paraphrased prompts."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "math_internal_logprob_probe_by_item.csv"
OUT_CSV = ROOT / "outputs" / "math_internal_trajectory_comparison.csv"
OUT_MD = ROOT / "outputs" / "math_internal_trajectory_summary.md"

NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
EXPR_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:\\times|[+\-*/^])\s*[-+]?\d+(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_RE = re.compile(r"(?:final answer|answer is|therefore|so,?)\s*[:=]?\s*([^.\n]{0,160})", re.I)
LET_RE = re.compile(r"\blet\s+([a-zA-Z])\b", re.I)
WORD_RE = re.compile(r"[a-zA-Z0-9]+")

OPERATION_PATTERNS = [
    ("probability", re.compile(r"\b(probability|binomial|choose|combinations?|outcomes?)\b", re.I)),
    ("geometry", re.compile(r"\b(area|triangle|cube|circle|radius|height|angle|distance|path|volume)\b", re.I)),
    ("algebra_equation", re.compile(r"\b(equation|solve|let\s+[a-zA-Z]|substitute|factor)\b", re.I)),
    ("multiplication", re.compile(r"(\\times|\bmultiply\b|\bproduct\b| \* )", re.I)),
    ("division", re.compile(r"(/|\bdivide\b|\bratio\b|\bfraction\b)", re.I)),
    ("addition", re.compile(r"(\+|\badd\b|\bsum\b|\btotal\b)", re.I)),
    ("subtraction", re.compile(r"(-|\bsubtract\b|\bdifference\b|\bremaining\b)", re.I)),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9./+\-*^=]", "", text)
    return text.lower()


def answer_like_value(text: str) -> str:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])
    expressions = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*(?:/|\+|-|\*|\^)\s*[-+]?\d+(?:\.\d+)?)*", text)
    if expressions:
        return normalize_answer(expressions[-1])
    return ""


def final_answer(text: str) -> str:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])
    finals = FINAL_RE.findall(text)
    if finals:
        extracted = answer_like_value(finals[-1])
        if extracted:
            return extracted
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])
    return ""


def first_operation(text: str) -> str:
    earliest = ("unknown", math.inf)
    for label, pattern in OPERATION_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < earliest[1]:
            earliest = (label, match.start())
    return earliest[0]


def first_numeric_expression(text: str) -> str:
    match = EXPR_RE.search(text)
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def variable_strategy(text: str) -> str:
    variables = LET_RE.findall(text)
    if variables:
        return ",".join(sorted(set(variable.lower() for variable in variables)))
    return ""


def numbers_first_window(text: str, window: int = 700) -> str:
    numbers = NUMBER_RE.findall(text[:window])
    return ";".join(numbers[:12])


def token_f1(left: str, right: str) -> float:
    left_tokens = [token.lower() for token in WORD_RE.findall(left)]
    right_tokens = [token.lower() for token in WORD_RE.findall(right)]
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum((left_counts & right_counts).values())
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_row(row: dict[str, str]) -> dict[str, object]:
    original = row["original_generated_text"]
    paraphrase = row["paraphrase_generated_text"]
    original_final = final_answer(original)
    paraphrase_final = final_answer(paraphrase)
    original_expr = first_numeric_expression(original)
    paraphrase_expr = first_numeric_expression(paraphrase)
    original_operation = first_operation(original)
    paraphrase_operation = first_operation(paraphrase)
    original_vars = variable_strategy(original)
    paraphrase_vars = variable_strategy(paraphrase)

    return {
        "item_id": row["item_id"],
        "selection_group": row["selection_group"],
        "mean_ncp": row["mean_ncp"],
        "generated_mean_logprob_delta": row["generated_mean_logprob_delta"],
        "original_operation": original_operation,
        "paraphrase_operation": paraphrase_operation,
        "operation_match": original_operation == paraphrase_operation,
        "original_first_numeric_expression": original_expr,
        "paraphrase_first_numeric_expression": paraphrase_expr,
        "first_numeric_expression_match": original_expr == paraphrase_expr,
        "original_variable_strategy": original_vars,
        "paraphrase_variable_strategy": paraphrase_vars,
        "variable_strategy_match": original_vars == paraphrase_vars,
        "original_early_numbers": numbers_first_window(original),
        "paraphrase_early_numbers": numbers_first_window(paraphrase),
        "early_numbers_match": numbers_first_window(original) == numbers_first_window(paraphrase),
        "original_final_answer": original_final,
        "paraphrase_final_answer": paraphrase_final,
        "final_answer_match": original_final == paraphrase_final and original_final != "",
        "output_token_f1": token_f1(original, paraphrase),
        "original_output_chars": len(original),
        "paraphrase_output_chars": len(paraphrase),
        "output_char_delta": len(paraphrase) - len(original),
    }


def summarize(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "# Math trajectory divergence probe",
        "",
        "Source: `outputs/math_internal_logprob_probe_by_item.csv` deterministic Together generations.",
        "",
        "## Validation",
        "",
        f"- Rows compared: {len(rows)}",
        f"- Unique items: {len({row['item_id'] for row in rows})}",
        f"- Rows with empty original final answer: {sum(not row['original_final_answer'] for row in rows)}",
        f"- Rows with empty paraphrase final answer: {sum(not row['paraphrase_final_answer'] for row in rows)}",
        "",
        "## Divergence summary",
        "",
    ]
    groups = [
        ("all", rows),
        ("shared_high", [row for row in rows if "cross_model_shared_high" in str(row["selection_group"])]),
        ("non_shared", [row for row in rows if "cross_model_shared_high" not in str(row["selection_group"])]),
    ]
    for name, group in groups:
        if not group:
            continue
        n = len(group)
        op_mismatch = sum(not row["operation_match"] for row in group)
        expr_mismatch = sum(not row["first_numeric_expression_match"] for row in group)
        final_mismatch = sum(not row["final_answer_match"] for row in group)
        early_num_mismatch = sum(not row["early_numbers_match"] for row in group)
        mean_f1 = sum(float(row["output_token_f1"]) for row in group) / n
        lines.append(
            f"- {name}: n={n}, operation mismatch={op_mismatch}, first numeric expression mismatch={expr_mismatch}, "
            f"early-number mismatch={early_num_mismatch}, final-answer mismatch={final_mismatch}, mean output F1={mean_f1:.3f}"
        )
    lines.extend(["", "## Strongest trajectory divergences", ""])
    for row in sorted(rows, key=lambda item: float(item["output_token_f1"]))[:10]:
        lines.append(
            f"- {row['item_id']}: F1={float(row['output_token_f1']):.3f}, "
            f"op={row['original_operation']}->{row['paraphrase_operation']}, "
            f"final={row['original_final_answer']}->{row['paraphrase_final_answer']}, "
            f"groups={row['selection_group']}"
        )
    return lines


def main() -> None:
    rows = [compare_row(row) for row in read_rows(INPUT)]
    write_csv(OUT_CSV, rows)
    OUT_MD.write_text("\n".join(summarize(rows)) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV} rows={len(rows)}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
