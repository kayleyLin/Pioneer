"""Trial reordering with resampling until the perturbed prompt changes.

This is a small-scope RQ1 utility. It tests the rule:

1. Sample a prompt from each task-type dataset.
2. Apply the current reordering perturbation.
3. Check whether the reordered prompt actually differs from the original.
4. If unchanged, reject that prompt and sample another prompt from the same task type.
5. Continue until each task type reaches the requested sample size or the attempt
   limit is reached.

The script does not overwrite the formal RQ1 prompt files.
"""

import argparse
import csv
import importlib.util
import random
import re
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SAMPLING_SCRIPT = ROOT / "src" / "06_sample_benchmark_prompts.py"
PERTURBATION_SCRIPT = ROOT / "src" / "19_create_rq1_perturbed_prompts.py"

DEFAULT_ORIGINAL_OUTPUT = ROOT / "prompts" / "rq1_reordering_resample_trial_original_prompts.csv"
DEFAULT_REORDERING_OUTPUT = ROOT / "prompts" / "rq1_reordering_resample_trial_prompts.csv"
DEFAULT_SUMMARY_OUTPUT = ROOT / "outputs" / "rq1_reordering_resample_trial_summary.csv"

SEED = 20260701


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampling = load_module(SAMPLING_SCRIPT, "rq1_sampling")
perturb = load_module(PERTURBATION_SCRIPT, "rq1_perturb")


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def changed_after_reordering(original: str, reordered: str) -> bool:
    return word_tokens(original) != word_tokens(reordered)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def fetch_rows(dataset_info: dict[str, str], offset: int, length: int) -> dict:
    query = {
        "dataset": dataset_info["dataset"],
        "config": dataset_info["config"],
        "split": dataset_info["split"],
        "offset": offset,
        "length": length,
    }
    url = f"{sampling.BASE_URL}?{urlencode(query)}"
    import json

    for retry in range(3):
        try:
            with urlopen(url, timeout=30, context=make_ssl_context()) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or retry == 2:
                raise
            time.sleep(5 * (retry + 1))

    raise RuntimeError("Unreachable retry state")


def get_total_rows(dataset_info: dict[str, str]) -> int:
    data = fetch_rows(dataset_info, offset=0, length=1)
    return int(data["num_rows_total"])


def get_row(dataset_info: dict[str, str], offset: int) -> dict:
    data = fetch_rows(dataset_info, offset=offset, length=1)
    return data["rows"][0]["row"]


def sample_task(
    task_type: str,
    k_per_task: int,
    max_attempts: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    dataset_info = sampling.DATASETS[task_type]
    total_rows = get_total_rows(dataset_info)

    accepted_originals: list[dict[str, str]] = []
    accepted_reordered: list[dict[str, str]] = []
    rejected_unchanged = 0
    attempts = 0
    used_offsets: set[int] = set()

    while len(accepted_originals) < k_per_task and attempts < max_attempts:
        offset = rng.randrange(total_rows)
        if offset in used_offsets:
            continue
        used_offsets.add(offset)

        length = min(batch_size, total_rows - offset)
        data = fetch_rows(dataset_info, offset=offset, length=length)

        for row_position, row_entry in enumerate(data["rows"]):
            if len(accepted_originals) >= k_per_task:
                break
            if attempts >= max_attempts:
                break

            source_index = int(row_entry.get("row_idx", offset + row_position))
            if source_index in used_offsets and source_index != offset:
                continue
            used_offsets.add(source_index)
            attempts += 1

            raw_row = row_entry["row"]
            original_row = sampling.normalize_row(task_type, source_index, raw_row)
            original_prompt = original_row["prompt_text"]
            reordered_prompt = perturb.reorder_prompt(original_prompt)

            if not changed_after_reordering(original_prompt, reordered_prompt):
                rejected_unchanged += 1
                continue

            accepted_originals.append(original_row)
            accepted_reordered.append(
                {
                    "item_id": original_row["item_id"],
                    "task_type": original_row["task_type"],
                    "dataset_name": original_row["dataset_name"],
                    "source_index": original_row["source_index"],
                    "perturbation_type": "reordering",
                    "construction_method": "rule_based_prompt_component_reordering",
                    "method_reference": "Reference 5 Haase et al.: information-order variation",
                    "validation_check": "token_sequence_non_identical",
                    "original_prompt": original_prompt,
                    "perturbed_prompt": perturb.clean_text(reordered_prompt),
                }
            )

        if len(accepted_originals) >= k_per_task:
            break

    summary = {
        "task_type": task_type,
        "target_valid_prompts": str(k_per_task),
        "accepted_valid_prompts": str(len(accepted_originals)),
        "attempts": str(attempts),
        "rejected_unchanged": str(rejected_unchanged),
        "status": "complete" if len(accepted_originals) == k_per_task else "incomplete",
    }
    return accepted_originals, accepted_reordered, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-per-task", type=int, default=5)
    parser.add_argument("--max-attempts-per-task", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--original-output", type=Path, default=DEFAULT_ORIGINAL_OUTPUT)
    parser.add_argument("--reordering-output", type=Path, default=DEFAULT_REORDERING_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()

    rng = random.Random(SEED)
    all_originals: list[dict[str, str]] = []
    all_reordered: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []

    for task_type in sampling.DATASETS:
        print(f"Sampling valid reordering prompts for {task_type}", flush=True)
        originals, reordered, summary = sample_task(
            task_type,
            args.k_per_task,
            args.max_attempts_per_task,
            args.batch_size,
            rng,
        )
        all_originals.extend(originals)
        all_reordered.extend(reordered)
        summaries.append(summary)
        print(
            f"  accepted={summary['accepted_valid_prompts']}/"
            f"{summary['target_valid_prompts']}, "
            f"attempts={summary['attempts']}, "
            f"rejected_unchanged={summary['rejected_unchanged']}, "
            f"status={summary['status']}",
            flush=True,
        )

    write_csv(
        args.original_output,
        all_originals,
        [
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
    write_csv(
        args.reordering_output,
        all_reordered,
        [
            "item_id",
            "task_type",
            "dataset_name",
            "source_index",
            "perturbation_type",
            "construction_method",
            "method_reference",
            "validation_check",
            "original_prompt",
            "perturbed_prompt",
        ],
    )
    write_csv(
        args.summary_output,
        summaries,
        [
            "task_type",
            "target_valid_prompts",
            "accepted_valid_prompts",
            "attempts",
            "rejected_unchanged",
            "status",
        ],
    )

    print(f"Wrote originals to {args.original_output}")
    print(f"Wrote reordering prompts to {args.reordering_output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
