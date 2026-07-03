"""Sample benchmark prompts for the RQ1 experiment.

The script uses Hugging Face datasets and performs stratified random sampling
by task type with a fixed seed.
"""

import argparse
import csv
import random
import re
import time
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "prompts" / "rq1_sampled_original_prompts.csv"
SEED = 20260623
K_PER_TASK = 10


DATASETS = {
    "factual_qa": {
        "dataset": "rajpurkar/squad_v2",
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
    "open_ended_writing": {
        "dataset": "tatsu-lab/alpaca",
        "config": "default",
        "split": "train",
    },
}


def load_split(dataset_info: dict[str, str]):
    for retry in range(3):
        try:
            return load_dataset(
                dataset_info["dataset"],
                dataset_info["config"],
                split=dataset_info["split"],
            )
        except Exception as error:
            if retry == 2:
                raise
            print(f"  retrying dataset load after {type(error).__name__}: {error}")
            time.sleep(5 * (retry + 1))

    raise RuntimeError("Unreachable retry state")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_boxed_answer(solution: str) -> str:
    match = re.search(r"\\boxed\{([^{}]+)\}", str(solution))
    if match:
        return clean_text(match.group(1))
    return clean_text(solution)


def normalize_row(
    task_type: str, source_index: int, row: dict, random_seed: int
) -> dict[str, str]:
    dataset_name = DATASETS[task_type]["dataset"]
    split = DATASETS[task_type]["split"]

    if task_type == "factual_qa":
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            raise ValueError("Skipping SQuAD V2 row without a reference answer")
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
    elif task_type == "open_ended_writing":
        instruction = row["instruction"]
        input_text = clean_text(row.get("input", ""))
        if not input_text:
            raise ValueError("Skipping Alpaca row without an input field")
        prompt = f"Instruction: {clean_text(instruction)}\n\nInput: {input_text}"
        reference = row.get("output", "")
        source_id = str(source_index)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-per-task", type=int, default=K_PER_TASK)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    rng = random.Random(args.seed)
    sampled_rows: list[dict[str, str]] = []

    for task_type, dataset_info in DATASETS.items():
        dataset = load_split(dataset_info)
        total_rows = len(dataset)
        sample_size = min(args.k_per_task, total_rows)
        sampled_offsets = rng.sample(range(total_rows), total_rows)

        print(
            f"{task_type}: sampling {sample_size} from "
            f"{dataset_info['dataset']} ({total_rows} rows)"
        )

        for offset in sampled_offsets:
            if sum(row["task_type"] == task_type for row in sampled_rows) >= sample_size:
                break
            raw_row = dataset[offset]
            try:
                sampled_rows.append(
                    normalize_row(task_type, offset, raw_row, args.seed)
                )
            except ValueError as error:
                print(f"  skipped row {offset}: {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "item_id",
                "task_type",
                "dataset_name",
                "dataset_split",
                "source_index",
                "source_id",
                "prompt_text",
                "reference_answer",
                "random_seed",
            ],
        )
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Wrote {len(sampled_rows)} sampled prompts to {output_path}")


if __name__ == "__main__":
    main()
