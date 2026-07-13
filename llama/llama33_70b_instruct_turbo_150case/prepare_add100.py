"""Prepare add100 prompts and shards for the Llama 3.3 70B branch."""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PROMPTS = BASE / "prompts" / "add100"
TMP = BASE / "tmp_parallel" / "add100"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_original(rows: list[dict[str, str]]) -> None:
    if len(rows) != 300:
        raise SystemExit(f"Expected 300 original prompts, found {len(rows)}")
    tasks = {"factual_qa": 0, "math_reasoning": 0, "code_generation": 0}
    for row in rows:
        tasks[row["task_type"]] += 1
    bad = {task: count for task, count in tasks.items() if count != 100}
    if bad:
        raise SystemExit(f"Original task counts invalid: {tasks}")
    if len({row["item_id"] for row in rows}) != 300:
        raise SystemExit("Original prompt item_ids are not unique")


def validate_perturbed(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1500:
        raise SystemExit(f"Expected 1500 perturbed prompts, found {len(rows)}")
    expected_perturbations = {
        "paraphrasing",
        "reordering",
        "formatting_changes",
        "context_injection",
        "surface_noise",
    }
    pairs = {(row["item_id"], row["perturbation_type"]) for row in rows}
    if len(pairs) != 1500:
        raise SystemExit("Perturbed prompt item_id+perturbation pairs are not unique")
    for task in ["factual_qa", "math_reasoning", "code_generation"]:
        task_rows = [row for row in rows if row["task_type"] == task]
        if len(task_rows) != 500:
            raise SystemExit(f"{task} perturbed prompt count invalid: {len(task_rows)}")
        by_item: dict[str, set[str]] = {}
        for row in task_rows:
            by_item.setdefault(row["item_id"], set()).add(row["perturbation_type"])
        if len(by_item) != 100:
            raise SystemExit(f"{task} item count invalid: {len(by_item)}")
        if any(values != expected_perturbations for values in by_item.values()):
            raise SystemExit(f"{task} perturbation set invalid")

    factual_para = [
        row
        for row in rows
        if row["task_type"] == "factual_qa"
        and row["perturbation_type"] == "paraphrasing"
    ]
    factual_bad = [
        row
        for row in factual_para
        if not (
            row["perturbed_prompt"].startswith("Context:")
            and "\nQuestion:" in row["perturbed_prompt"]
        )
    ]
    if factual_bad:
        raise SystemExit(f"Factual QA paraphrasing malformed rows: {len(factual_bad)}")

    artifact_re = re.compile(
        r"(Rewrite\s+the\s+prompt|Research Question:|Code Signature:|Task Signature:|```)",
        re.IGNORECASE,
    )
    math_para = [
        row
        for row in rows
        if row["task_type"] == "math_reasoning"
        and row["perturbation_type"] == "paraphrasing"
    ]
    math_art = [row for row in math_para if artifact_re.search(row["perturbed_prompt"])]
    if math_art:
        raise SystemExit(f"Math paraphrasing template artifacts: {len(math_art)}")
    asy_removed = [
        row
        for row in math_para
        if "[asy]" in row["original_prompt"] and "[asy]" not in row["perturbed_prompt"]
    ]
    if asy_removed:
        raise SystemExit(f"Math paraphrasing ASY removed rows: {len(asy_removed)}")


def shard(rows: list[dict[str, str]], n: int) -> list[list[dict[str, str]]]:
    buckets = [[] for _ in range(n)]
    for index, row in enumerate(rows):
        buckets[index % n].append(row)
    return buckets


def main() -> None:
    original_source = ROOT / "prompts" / "rq1_sampled_original_prompts_add100_three_task.csv"
    perturbed_sources = [
        ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_factual_qa.csv",
        ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_math_reasoning.csv",
        ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_code_generation.csv",
    ]
    original_rows = read_csv(original_source)
    perturbed_rows: list[dict[str, str]] = []
    for path in perturbed_sources:
        perturbed_rows.extend(read_csv(path))

    validate_original(original_rows)
    validate_perturbed(perturbed_rows)

    write_csv(
        PROMPTS / "original_prompts_add100_three_task.csv",
        original_rows,
        list(original_rows[0].keys()),
    )
    write_csv(
        PROMPTS / "perturbed_prompts_add100_three_task.csv",
        perturbed_rows,
        list(perturbed_rows[0].keys()),
    )
    for index, bucket in enumerate(shard(original_rows, 5), start=1):
        write_csv(TMP / f"original_prompts_shard{index}.csv", bucket, list(original_rows[0].keys()))
    for index, bucket in enumerate(shard(perturbed_rows, 5), start=1):
        write_csv(TMP / f"perturbed_prompts_shard{index}.csv", bucket, list(perturbed_rows[0].keys()))

    print("Prepared add100 prompts and 5 shards.")
    print("Original prompts: 300")
    print("Perturbed prompts: 1500")
    print("Expected original generations per shard: 300")
    print("Expected perturbed generations per shard: 1500")


if __name__ == "__main__":
    main()
