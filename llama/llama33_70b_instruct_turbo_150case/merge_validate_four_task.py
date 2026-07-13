"""Merge the Llama 3.3 70B n50 four-task outputs and validate them."""

import csv
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
PROMPTS = BASE / "prompts"


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
    if len(rows) != 1000:
        problems.append(f"original rows: {len(rows)} != 1000")
    if len({row["item_id"] for row in rows}) != 200:
        problems.append("original cases != 200")
    if len(keys) != len(set(keys)):
        problems.append("original duplicate keys found")
    if any(not row.get("output_text", "").strip() for row in rows):
        problems.append("original empty outputs found")
    for task in ["code_generation", "factual_qa", "math_reasoning", "open_ended_writing"]:
        if sum(1 for row in rows if row["task_type"] == task) != 250:
            problems.append(f"original {task} rows != 250")
    samples: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        samples[row["item_id"]].add(row["sample_id"])
        prompt = prompt_map.get(row["item_id"])
        if prompt is None or row["prompt_text"] != prompt["prompt_text"]:
            problems.append(f"original prompt mismatch: {row['item_id']}")
            break
    expected_samples = {"1", "2", "3", "4", "5"}
    if any(value != expected_samples for value in samples.values()):
        problems.append("original bad sample sets found")
    return problems


def validate_perturbed(rows: list[dict[str, str]], prompt_rows: list[dict[str, str]]) -> list[str]:
    problems: list[str] = []
    prompt_map = {
        (row["item_id"], row["perturbation_type"]): row for row in prompt_rows
    }
    keys = [(row["item_id"], row["perturbation_type"], row["sample_id"]) for row in rows]
    if len(rows) != 5000:
        problems.append(f"perturbed rows: {len(rows)} != 5000")
    if len({row["item_id"] for row in rows}) != 200:
        problems.append("perturbed cases != 200")
    if len(keys) != len(set(keys)):
        problems.append("perturbed duplicate keys found")
    if any(not row.get("output_text", "").strip() for row in rows):
        problems.append("perturbed empty outputs found")
    for task in ["code_generation", "factual_qa", "math_reasoning", "open_ended_writing"]:
        if sum(1 for row in rows if row["task_type"] == task) != 1250:
            problems.append(f"perturbed {task} rows != 1250")
    for perturbation in [
        "context_injection",
        "formatting_changes",
        "paraphrasing",
        "reordering",
        "surface_noise",
    ]:
        if sum(1 for row in rows if row["perturbation_type"] == perturbation) != 1000:
            problems.append(f"perturbed {perturbation} rows != 1000")
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
    expected_samples = {"1", "2", "3", "4", "5"}
    if any(value != expected_samples for value in samples.values()):
        problems.append("perturbed bad sample sets found")
    return problems


def main() -> None:
    original_three = read_csv(
        OUT / "rq1_llama33_70b_original_generations_150case_three_task.csv"
    )
    original_open = read_csv(OUT / "original_generations_n50_open_ended_writing.csv")
    perturbed_three = read_csv(
        OUT / "rq1_llama33_70b_perturbed_generations_150case_three_task.csv"
    )
    perturbed_open = read_csv(OUT / "perturbed_generations_n50_open_ended_writing.csv")

    original_rows = original_three + original_open
    perturbed_rows = perturbed_three + perturbed_open
    original_rows.sort(key=lambda row: (row["task_type"], row["item_id"], int(row["sample_id"])))
    perturbed_rows.sort(
        key=lambda row: (
            row["task_type"],
            row["item_id"],
            row["perturbation_type"],
            int(row["sample_id"]),
        )
    )

    original_prompts = read_csv(PROMPTS / "original_prompts_150case_three_task.csv") + read_csv(
        PROMPTS / "original_prompts_n50_open_ended_writing.csv"
    )
    perturbed_prompts = read_csv(PROMPTS / "perturbed_prompts_150case_three_task.csv") + read_csv(
        PROMPTS / "perturbed_prompts_n50_open_ended_writing.csv"
    )

    original_problems = validate_original(original_rows, original_prompts)
    perturbed_problems = validate_perturbed(perturbed_rows, perturbed_prompts)
    problems = original_problems + perturbed_problems
    if problems:
        raise SystemExit("Validation failed:\n" + "\n".join(f"- {p}" for p in problems))

    write_csv(
        OUT / "rq1_llama33_70b_original_generations_n50_four_task.csv",
        original_rows,
        list(original_rows[0].keys()),
    )
    write_csv(
        OUT / "rq1_llama33_70b_perturbed_generations_n50_four_task.csv",
        perturbed_rows,
        list(perturbed_rows[0].keys()),
    )

    report = [
        "# Llama 3.3 70B n50 four-task validation report",
        "",
        "## Validation",
        "",
        "- Tasks: code_generation, factual_qa, math_reasoning, open_ended_writing",
        "- Expected items per task: 50",
        "- Expected samples per prompt: 5",
        f"- Original rows: {len(original_rows)}",
        f"- Perturbed rows: {len(perturbed_rows)}",
        "- Duplicate keys: none",
        "- Empty outputs: none",
        "- Prompt mismatch: none",
        "- Original task counts: " + str(dict(Counter(row["task_type"] for row in original_rows))),
        "- Perturbed task counts: " + str(dict(Counter(row["task_type"] for row in perturbed_rows))),
        "- Perturbation counts: "
        + str(dict(Counter(row["perturbation_type"] for row in perturbed_rows))),
        "",
        "## Status",
        "",
        "PASS",
    ]
    (OUT / "generation_validation_report_n50_four_task.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("Wrote four-task merged outputs and validation report.")


if __name__ == "__main__":
    main()
