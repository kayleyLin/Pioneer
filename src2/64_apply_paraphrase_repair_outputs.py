"""Apply repaired paraphrasing generations and validate prompt consistency."""

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPAIR_DIR = OUTPUTS / "paraphrase_repair_outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["item_id"], row["perturbation_type"], row["sample_id"])


def prompt_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["item_id"], row["perturbation_type"])


def load_prompt_map(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    prompt_map = {}
    for path in paths:
        for row in read_csv(path):
            prompt_map[prompt_key(row)] = row
    return prompt_map


def load_repair_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(REPAIR_DIR.glob(pattern)):
        rows.extend(read_csv(path))
    return rows


def validate_rows(
    rows: list[dict[str, str]],
    prompt_map: dict[tuple[str, str], dict[str, str]],
    expected_rows: int,
    expected_cases_by_task: dict[str, int],
    expected_samples: int,
) -> list[str]:
    problems: list[str] = []
    if len(rows) != expected_rows:
        problems.append(f"row count mismatch: {len(rows)} != {expected_rows}")

    duplicate_count = len(rows) - len({key(row) for row in rows})
    if duplicate_count:
        problems.append(f"duplicate generation keys: {duplicate_count}")

    empty_count = sum(1 for row in rows if not row.get("output_text", "").strip())
    if empty_count:
        problems.append(f"empty outputs: {empty_count}")

    for task, expected_cases in expected_cases_by_task.items():
        cases = {row["item_id"] for row in rows if row["task_type"] == task}
        if len(cases) != expected_cases:
            problems.append(f"{task} case count mismatch: {len(cases)} != {expected_cases}")

    sample_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        sample_sets[prompt_key(row)].add(row["sample_id"])
    expected_sample_set = {str(i) for i in range(1, expected_samples + 1)}
    bad_samples = [
        (pair, sorted(samples))
        for pair, samples in sample_sets.items()
        if samples != expected_sample_set
    ]
    if bad_samples:
        problems.append(f"bad sample sets: {len(bad_samples)}")

    prompt_mismatches = []
    for row in rows:
        prompt = prompt_map.get(prompt_key(row))
        if prompt is None:
            prompt_mismatches.append((prompt_key(row), "missing current prompt"))
            continue
        if row.get("perturbed_prompt") != prompt.get("perturbed_prompt"):
            prompt_mismatches.append((prompt_key(row), "perturbed_prompt mismatch"))
        if row.get("original_prompt") != prompt.get("original_prompt"):
            prompt_mismatches.append((prompt_key(row), "original_prompt mismatch"))
    if prompt_mismatches:
        unique = sorted(set(prompt_mismatches))
        problems.append(f"prompt mismatches: {len(unique)} unique")

    return problems


def write_report(path: Path, title: str, rows: list[dict[str, str]], problems: list[str]) -> None:
    task_counts = Counter(row["task_type"] for row in rows)
    perturbation_counts = Counter(row["perturbation_type"] for row in rows)
    model_counts = Counter(row["model_name"] for row in rows)
    lines = [
        f"# {title}",
        "",
        "## Counts",
        "",
        f"- Rows: {len(rows)}",
        f"- Tasks: {dict(task_counts)}",
        f"- Perturbations: {dict(perturbation_counts)}",
        f"- Models: {dict(model_counts)}",
        f"- Empty outputs: {sum(1 for row in rows if not row.get('output_text', '').strip())}",
        f"- Duplicate keys: {len(rows) - len({key(row) for row in rows})}",
        "",
        "## Status",
        "",
    ]
    if problems:
        lines.append("FAIL")
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("PASS")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    add100_path = OUTPUTS / "rq1_formal_perturbed_generations_add100_three_task.csv"
    n50_math_path = OUTPUTS / "rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv"
    n50_math_para_path = (
        OUTPUTS / "rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv"
    )

    add100_rows = read_csv(add100_path)
    add100_fields = list(add100_rows[0].keys())
    repair_rows = load_repair_rows("add100_paraphrasing_repair_shard*.csv")
    repair_rows = [{field: row.get(field, "") for field in add100_fields} for row in repair_rows]
    repair_keys = {key(row) for row in repair_rows}
    if len(repair_rows) != 1000 or len(repair_keys) != 1000:
        raise SystemExit(
            f"Expected 1000 unique add100 repair rows, got rows={len(repair_rows)} "
            f"keys={len(repair_keys)}"
        )

    replacement_scopes = {
        ("factual_qa", "paraphrasing"),
        ("math_reasoning", "paraphrasing"),
    }
    kept_add100 = [
        row
        for row in add100_rows
        if (row["task_type"], row["perturbation_type"]) not in replacement_scopes
    ]
    repaired_add100 = kept_add100 + repair_rows
    repaired_add100.sort(key=lambda row: (row["task_type"], row["item_id"], row["perturbation_type"], int(row["sample_id"])))

    add100_prompt_map = load_prompt_map(
        [
            ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_factual_qa.csv",
            ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_math_reasoning.csv",
            ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_code_generation.csv",
        ]
    )
    add100_problems = validate_rows(
        repaired_add100,
        add100_prompt_map,
        expected_rows=7500,
        expected_cases_by_task={
            "factual_qa": 100,
            "math_reasoning": 100,
            "code_generation": 100,
        },
        expected_samples=5,
    )
    if add100_problems:
        write_report(
            OUTPUTS / "add100_generation_validation_report.md",
            "add100 generation merge validation report",
            repaired_add100,
            add100_problems,
        )
        raise SystemExit("add100 validation failed: " + "; ".join(add100_problems))
    write_csv(add100_path, repaired_add100, add100_fields)
    write_report(
        OUTPUTS / "add100_generation_validation_report.md",
        "add100 generation merge validation report",
        repaired_add100,
        add100_problems,
    )

    n50_repair = load_repair_rows("n50_math_5201_paraphrasing.csv")
    n50_repair = [{field: row.get(field, "") for field in add100_fields} for row in n50_repair]
    if len(n50_repair) != 5 or len({key(row) for row in n50_repair}) != 5:
        raise SystemExit(f"Expected 5 unique n50 math repair rows, got {len(n50_repair)}")

    n50_rows = read_csv(n50_math_path)
    n50_fields = list(n50_rows[0].keys())
    n50_repair = [{field: row.get(field, "") for field in n50_fields} for row in n50_repair]
    n50_keys = {key(row) for row in n50_repair}
    repaired_n50 = [row for row in n50_rows if key(row) not in n50_keys] + n50_repair
    repaired_n50.sort(key=lambda row: (row["item_id"], row["perturbation_type"], int(row["sample_id"])))

    n50_prompt_map = load_prompt_map(
        [ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv"]
    )
    n50_problems = validate_rows(
        repaired_n50,
        n50_prompt_map,
        expected_rows=1250,
        expected_cases_by_task={"math_reasoning": 50},
        expected_samples=5,
    )
    if n50_problems:
        raise SystemExit("n50 math validation failed: " + "; ".join(n50_problems))
    write_csv(n50_math_path, repaired_n50, n50_fields)

    n50_para_rows = read_csv(n50_math_para_path)
    n50_para_fields = list(n50_para_rows[0].keys())
    repaired_n50_para = [row for row in n50_para_rows if key(row) not in n50_keys] + [
        {field: row.get(field, "") for field in n50_para_fields} for row in n50_repair
    ]
    repaired_n50_para.sort(key=lambda row: (row["item_id"], row["perturbation_type"], int(row["sample_id"])))
    para_problems = validate_rows(
        repaired_n50_para,
        n50_prompt_map,
        expected_rows=250,
        expected_cases_by_task={"math_reasoning": 50},
        expected_samples=5,
    )
    if para_problems:
        raise SystemExit("n50 math paraphrasing validation failed: " + "; ".join(para_problems))
    write_csv(n50_math_para_path, repaired_n50_para, n50_para_fields)

    write_report(
        OUTPUTS / "n50_math_5201_repair_validation_report.md",
        "n50 math_reasoning_5201 repair validation report",
        repaired_n50_para,
        para_problems,
    )
    print("Applied paraphrasing repairs and validation passed.")


if __name__ == "__main__":
    main()
