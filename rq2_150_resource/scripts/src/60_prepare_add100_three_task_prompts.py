"""Prepare add100 original prompt files for three RQ1 tasks.

This script samples 100 additional items for:
    factual_qa, math_reasoning, code_generation

It excludes the existing n50 prompt rows and does not include
open_ended_writing. The output files are prompt CSVs only; no model calls are
made here.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260623
ITEMS_PER_TASK = 100
TASKS = ["factual_qa", "math_reasoning", "code_generation"]

DATASETS = {
    "factual_qa": {
        "dataset": "rajpurkar/squad_v2",
        "load_dataset": "squad_v2",
        "config": "squad_v2",
        "split": "validation",
    },
    "math_reasoning": {
        "dataset": "nlile/hendrycks-MATH-benchmark",
        "config": "default",
        "split": "train",
    },
    "code_generation": {
        "dataset": "bigcode/humanevalpack",
        "config": "python",
        "split": "test",
    },
}

FIELDNAMES = [
    "item_id",
    "task_type",
    "dataset_name",
    "dataset_split",
    "source_index",
    "source_id",
    "prompt_text",
    "reference_answer",
    "random_seed",
]


def clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_boxed_answer(solution: str) -> str:
    match = re.search(r"\\boxed\{([^{}]+)\}", str(solution))
    if match:
        return clean_text(match.group(1))
    return clean_text(solution)


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists() or (path.parts and path.parts[0].lower() == "pioneer"):
        return path.resolve()
    return ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_split(dataset_info: dict[str, str]):
    from datasets import load_dataset

    for retry in range(3):
        try:
            dataset_path = dataset_info.get("load_dataset", dataset_info["dataset"])
            config = dataset_info.get("config")
            if dataset_path == "squad_v2":
                config = None
            if config:
                return load_dataset(dataset_path, config, split=dataset_info["split"])
            return load_dataset(dataset_path, split=dataset_info["split"])
        except Exception as error:
            if retry == 2:
                raise
            print(f"  retrying dataset load after {type(error).__name__}: {error}")
            time.sleep(5 * (retry + 1))
    raise RuntimeError("unreachable retry state")


def normalize_row(
    task_type: str, source_index: int, row: dict, random_seed: int
) -> dict[str, str]:
    dataset_name = DATASETS[task_type]["dataset"]
    split = DATASETS[task_type]["split"]

    if task_type == "factual_qa":
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            raise ValueError("SQuAD V2 row has no reference answer")
        context = clean_text(row["context"])
        question = clean_text(row["question"])
        prompt = f"Context: {context}\n\nQuestion: {question}"
        reference = answers[0]
        source_id = row.get("id", str(source_index))
    elif task_type == "math_reasoning":
        problem = clean_text(row["problem"])
        prompt = (
            f"Problem: {problem}\n\n"
            "Instruction: Solve the problem and provide the final answer."
        )
        reference = row.get("answer") or extract_boxed_answer(row["solution"])
        source_id = row.get("unique_id", str(source_index))
    elif task_type == "code_generation":
        prompt = row.get("instruction") or row["prompt"]
        reference = row["canonical_solution"]
        source_id = row.get("task_id", str(source_index))
    else:
        raise ValueError(f"unsupported task_type: {task_type}")

    return {
        "item_id": f"{task_type}_{source_index}",
        "task_type": task_type,
        "dataset_name": dataset_name,
        "dataset_split": split,
        "source_index": str(source_index),
        "source_id": clean_text(source_id),
        "prompt_text": prompt.strip(),
        "reference_answer": clean_text(reference),
        "random_seed": str(random_seed),
    }


def existing_keys(existing_files: list[Path]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for path in existing_files:
        if not path.exists():
            raise SystemExit(f"Missing existing prompt file: {path}")
        for row in read_csv(path):
            keys.add((row["task_type"], row["source_index"], row["source_id"]))
    return keys


def sample_task(
    task_type: str,
    rng: random.Random,
    count: int,
    existing: set[tuple[str, str, str]],
    seed: int,
) -> list[dict[str, str]]:
    dataset = load_split(DATASETS[task_type])
    offsets = rng.sample(range(len(dataset)), len(dataset))
    rows: list[dict[str, str]] = []
    skipped_existing = 0
    skipped_invalid = 0

    print(f"{task_type}: sampling {count} add100 rows from {len(dataset)} records")
    for offset in offsets:
        try:
            row = normalize_row(task_type, offset, dataset[offset], seed)
        except ValueError:
            skipped_invalid += 1
            continue

        key = (row["task_type"], row["source_index"], row["source_id"])
        if key in existing:
            skipped_existing += 1
            continue

        rows.append(row)
        if len(rows) == count:
            break

    if len(rows) != count:
        raise SystemExit(
            f"{task_type}: only sampled {len(rows)} rows; needed {count}. "
            f"skipped_existing={skipped_existing}, skipped_invalid={skipped_invalid}"
        )

    print(
        f"{task_type}: sampled {len(rows)} rows; "
        f"skipped_existing={skipped_existing}; skipped_invalid={skipped_invalid}"
    )
    return rows


def validate(rows_by_task: dict[str, list[dict[str, str]]], count: int) -> None:
    all_rows = [row for rows in rows_by_task.values() for row in rows]
    if any(row["task_type"] == "open_ended_writing" for row in all_rows):
        raise SystemExit("open_ended_writing rows are not allowed in add100")

    for task_type in TASKS:
        rows = rows_by_task.get(task_type, [])
        if len(rows) != count:
            raise SystemExit(f"{task_type}: expected {count} rows, got {len(rows)}")

    item_ids = [row["item_id"] for row in all_rows]
    if len(item_ids) != len(set(item_ids)):
        raise SystemExit("Duplicate item_id values found in add100 rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-per-task", type=int, default=ITEMS_PER_TASK)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--existing-prompts",
        nargs="+",
        type=Path,
        default=[
            Path("prompts/rq1_sampled_original_prompts_n50_factual_qa.csv"),
            Path("prompts/rq1_sampled_original_prompts_n50_math_reasoning.csv"),
            Path("prompts/rq1_sampled_original_prompts_n50_code_generation.csv"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompts"),
        help="Directory for per-task and combined add100 prompt CSVs.",
    )
    parser.add_argument(
        "--tag",
        default="add100",
        help="Filename tag used in rq1_sampled_original_prompts_<tag>_*.csv.",
    )
    args = parser.parse_args()

    existing = existing_keys([resolve(path) for path in args.existing_prompts])
    output_dir = resolve(args.output_dir)
    rng = random.Random(args.seed)

    rows_by_task = {
        task_type: sample_task(
            task_type,
            rng,
            args.items_per_task,
            existing,
            args.seed,
        )
        for task_type in TASKS
    }
    validate(rows_by_task, args.items_per_task)

    combined: list[dict[str, str]] = []
    for task_type in TASKS:
        rows = rows_by_task[task_type]
        combined.extend(rows)
        out = output_dir / f"rq1_sampled_original_prompts_{args.tag}_{task_type}.csv"
        write_csv(out, rows)
        print(f"Wrote {len(rows)} rows: {out}")

    combined_path = output_dir / f"rq1_sampled_original_prompts_{args.tag}_three_task.csv"
    write_csv(combined_path, combined)
    print(f"Wrote {len(combined)} combined rows: {combined_path}")


if __name__ == "__main__":
    main()
