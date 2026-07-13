"""Merge and validate add100 outputs for the Llama 3.3 70B branch."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "add100"
PROMPTS = BASE / "prompts" / "add100"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_original(rows: list[dict[str, str]], prompt_rows: list[dict[str, str]]) -> list[str]:
    problems: list[str] = []
    prompt_map = {row["item_id"]: row for row in prompt_rows}
    keys = [(row["item_id"], row["sample_id"]) for row in rows]
    if len(rows) != 1500:
        problems.append(f"original rows {len(rows)} != 1500")
    if len({row["item_id"] for row in rows}) != 300:
        problems.append("original cases != 300")
    if len(keys) != len(set(keys)):
        problems.append("original duplicate keys")
    if any(not row.get("output_text", "").strip() for row in rows):
        problems.append("original empty outputs")
    for task in ["code_generation", "factual_qa", "math_reasoning"]:
        if sum(1 for row in rows if row["task_type"] == task) != 500:
            problems.append(f"original {task} rows != 500")
    samples: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        samples[row["item_id"]].add(row["sample_id"])
        prompt = prompt_map.get(row["item_id"])
        if prompt is None or row["prompt_text"] != prompt["prompt_text"]:
            problems.append(f"original prompt mismatch: {row['item_id']}")
            break
    if any(value != {"1", "2", "3", "4", "5"} for value in samples.values()):
        problems.append("original bad sample sets")
    return problems


def validate_perturbed(rows: list[dict[str, str]], prompt_rows: list[dict[str, str]]) -> list[str]:
    problems: list[str] = []
    prompt_map = {
        (row["item_id"], row["perturbation_type"]): row for row in prompt_rows
    }
    keys = [(row["item_id"], row["perturbation_type"], row["sample_id"]) for row in rows]
    if len(rows) != 7500:
        problems.append(f"perturbed rows {len(rows)} != 7500")
    if len({row["item_id"] for row in rows}) != 300:
        problems.append("perturbed cases != 300")
    if len(keys) != len(set(keys)):
        problems.append("perturbed duplicate keys")
    if any(not row.get("output_text", "").strip() for row in rows):
        problems.append("perturbed empty outputs")
    for task in ["code_generation", "factual_qa", "math_reasoning"]:
        if sum(1 for row in rows if row["task_type"] == task) != 2500:
            problems.append(f"perturbed {task} rows != 2500")
    for perturbation in [
        "context_injection",
        "formatting_changes",
        "paraphrasing",
        "reordering",
        "surface_noise",
    ]:
        if sum(1 for row in rows if row["perturbation_type"] == perturbation) != 1500:
            problems.append(f"perturbed {perturbation} rows != 1500")
    samples: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        pair = (row["item_id"], row["perturbation_type"])
        samples[pair].add(row["sample_id"])
        prompt = prompt_map.get(pair)
        if (
            prompt is None
            or row["original_prompt"] != prompt["original_prompt"]
            or row["perturbed_prompt"] != prompt["perturbed_prompt"]
        ):
            problems.append(f"perturbed prompt mismatch: {pair}")
            break
    if any(value != {"1", "2", "3", "4", "5"} for value in samples.values()):
        problems.append("perturbed bad sample sets")

    factual_bad = [
        row
        for row in rows
        if row["task_type"] == "factual_qa"
        and row["perturbation_type"] == "paraphrasing"
        and not (
            row["perturbed_prompt"].startswith("Context:")
            and "\nQuestion:" in row["perturbed_prompt"]
        )
    ]
    if factual_bad:
        problems.append(f"factual paraphrasing malformed rows: {len(factual_bad)}")
    artifact_re = re.compile(
        r"(Rewrite\s+the\s+prompt|Research Question:|Code Signature:|Task Signature:|```)",
        re.IGNORECASE,
    )
    math_art = [
        row
        for row in rows
        if row["task_type"] == "math_reasoning"
        and row["perturbation_type"] == "paraphrasing"
        and artifact_re.search(row["perturbed_prompt"])
    ]
    if math_art:
        problems.append(f"math template artifact rows: {len(math_art)}")
    asy_removed = [
        row
        for row in rows
        if row["task_type"] == "math_reasoning"
        and row["perturbation_type"] == "paraphrasing"
        and "[asy]" in row["original_prompt"]
        and "[asy]" not in row["perturbed_prompt"]
    ]
    if asy_removed:
        problems.append(f"math ASY removed rows: {len(asy_removed)}")
    return problems


def main() -> None:
    original_rows: list[dict[str, str]] = []
    perturbed_rows: list[dict[str, str]] = []
    for path in sorted(OUT.glob("original_generations_shard*.csv")):
        original_rows.extend(read_csv(path))
    for path in sorted(OUT.glob("perturbed_generations_shard*.csv")):
        perturbed_rows.extend(read_csv(path))

    original_rows.sort(key=lambda row: (row["task_type"], row["item_id"], int(row["sample_id"])))
    perturbed_rows.sort(
        key=lambda row: (
            row["task_type"],
            row["item_id"],
            row["perturbation_type"],
            int(row["sample_id"]),
        )
    )

    original_prompts = read_csv(PROMPTS / "original_prompts_add100_three_task.csv")
    perturbed_prompts = read_csv(PROMPTS / "perturbed_prompts_add100_three_task.csv")
    problems = validate_original(original_rows, original_prompts) + validate_perturbed(
        perturbed_rows, perturbed_prompts
    )
    report = [
        "# Llama 3.3 70B add100 three-task validation report",
        "",
        "## Validation",
        "",
        "- Tasks: code_generation, factual_qa, math_reasoning",
        "- Expected items per task: 100",
        "- Expected samples per prompt: 5",
        f"- Original rows: {len(original_rows)}",
        f"- Perturbed rows: {len(perturbed_rows)}",
        f"- Original task counts: {dict(Counter(row['task_type'] for row in original_rows))}",
        f"- Perturbed task counts: {dict(Counter(row['task_type'] for row in perturbed_rows))}",
        f"- Perturbation counts: {dict(Counter(row['perturbation_type'] for row in perturbed_rows))}",
        "",
        "## Status",
        "",
        "PASS" if not problems else "FAIL",
    ]
    report.extend(f"- {problem}" for problem in problems)
    (OUT / "generation_validation_report_add100_three_task.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if problems:
        raise SystemExit("Validation failed:\n" + "\n".join(problems))

    write_csv(
        OUT / "rq1_llama33_70b_original_generations_add100_three_task.csv",
        original_rows,
        list(original_rows[0].keys()),
    )
    write_csv(
        OUT / "rq1_llama33_70b_perturbed_generations_add100_three_task.csv",
        perturbed_rows,
        list(perturbed_rows[0].keys()),
    )
    print("Wrote add100 merged outputs and validation report.")


if __name__ == "__main__":
    main()
