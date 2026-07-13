"""Generate repeated RQ1 outputs with the Qwen (DashScope) API.

Before running:

    export DASHSCOPE_API_KEY="your_api_key_here"
    python3 src/07_generate_rq1_outputs_qwen.py

This script reads prompts/rq1_sampled_original_prompts_n50.csv and writes
qwen/outputs/rq1_qwen_original_generations_n50.csv by default.

Per-task output:

    python3 src/07_generate_rq1_outputs_qwen.py \
        --input prompts/rq1_sampled_original_prompts_n50_factual_qa.csv \
        --output qwen/outputs/rq1_qwen_original_generations_n50_factual_qa.csv
"""

import argparse
import csv
import importlib.util
import json
import os
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50.csv"
OUTPUTS = ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50.csv"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-plus"
FIELDNAMES = [
    "item_id",
    "task_type",
    "dataset_name",
    "source_index",
    "sample_id",
    "model_name",
    "temperature",
    "top_p",
    "max_output_tokens",
    "prompt_text",
    "output_text",
]


def make_ssl_context() -> ssl.SSLContext | None:
    if importlib.util.find_spec("certifi") is None:
        return None
    import certifi
    return ssl.create_default_context(cafile=certifi.where())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    rows = read_csv(path)
    return {(row["item_id"], row["sample_id"]) for row in rows}


def append_row(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def extract_output_text(response_data: dict) -> str:
    """Extract text from Qwen chat/completions response."""
    return response_data["choices"][0]["message"]["content"].strip()


def call_qwen(prompt: str, config: dict, api_key: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": config["system_prompt"],
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(1, 6):
        try:
            with urlopen(request, timeout=120, context=make_ssl_context()) as response:
                response_data = json.load(response)
            break
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 5:
                raise RuntimeError(f"Qwen API error {error.code}: {details}") from error
            wait_seconds = 5 * attempt
            print(
                f"Qwen API error {error.code}; retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        except (RemoteDisconnected, TimeoutError, URLError) as error:
            if attempt == 5:
                raise RuntimeError(
                    f"Qwen connection error after {attempt} attempts: {error}"
                ) from error
            wait_seconds = 5 * attempt
            print(
                f"Connection error: {type(error).__name__}; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)

    output_text = extract_output_text(response_data)
    if not output_text:
        raise RuntimeError(f"No output_text found in response: {response_data}")
    return output_text


def make_output_row(
    prompt_row: dict[str, str],
    sample_id: int,
    output_text: str,
    config: dict,
) -> dict[str, str | int | float]:
    return {
        "item_id": prompt_row["item_id"],
        "task_type": prompt_row["task_type"],
        "dataset_name": prompt_row["dataset_name"],
        "source_index": prompt_row["source_index"],
        "sample_id": sample_id,
        "model_name": MODEL,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_output_tokens": config.get("max_output_tokens", ""),
        "prompt_text": prompt_row["prompt_text"],
        "output_text": output_text,
    }


def run_generation_job(
    job_number: int,
    total: int,
    prompt_row: dict[str, str],
    sample_id: int,
    config: dict,
    api_key: str,
) -> tuple[int, dict[str, str | int | float]]:
    print(
        f"[{job_number}/{total}] {prompt_row['item_id']} sample {sample_id}",
        flush=True,
    )
    output_text = call_qwen(prompt_row["prompt_text"], config, api_key)
    return job_number, make_output_row(prompt_row, sample_id, output_text, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROMPTS)
    parser.add_argument("--output", type=Path, default=OUTPUTS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--limit-prompts", type=int)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.samples is not None and args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.limit_prompts is not None and args.limit_prompts < 1:
        raise SystemExit("--limit-prompts must be at least 1")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "DASHSCOPE_API_KEY is not set. Run:\n"
            'export DASHSCOPE_API_KEY="your_api_key_here"\n'
            "then run this script again."
        )

    config = read_config(CONFIG)
    prompts_path = resolve_path(args.input)
    outputs_path = resolve_path(args.output)
    prompts = read_csv(prompts_path)
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    n_samples = args.samples or int(config["n_samples_per_prompt"])

    total = len(prompts) * n_samples
    seen = existing_keys(outputs_path)
    if seen:
        print(f"Found {len(seen)} existing rows in {outputs_path}; resuming.")
    jobs: list[tuple[int, dict[str, str], int]] = []
    job_number = 0

    for prompt_row in prompts:
        for sample_id in range(1, n_samples + 1):
            job_number += 1
            key = (prompt_row["item_id"], str(sample_id))
            if key in seen:
                print(
                    f"[{job_number}/{total}] "
                    f"{prompt_row['item_id']} sample {sample_id} skipped"
                )
                continue
            jobs.append((job_number, prompt_row, sample_id))

    print(f"Submitting {len(jobs)} jobs with workers={args.workers}.")
    if args.workers == 1:
        for job_number, prompt_row, sample_id in jobs:
            _, row = run_generation_job(
                job_number, total, prompt_row, sample_id, config, api_key
            )
            append_row(outputs_path, row)
            seen.add((str(row["item_id"]), str(row["sample_id"])))
            time.sleep(0.2)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_generation_job,
                    job_number,
                    total,
                    prompt_row,
                    sample_id,
                    config,
                    api_key,
                )
                for job_number, prompt_row, sample_id in jobs
            ]
            for future in as_completed(futures):
                _, row = future.result()
                append_row(outputs_path, row)
                seen.add((str(row["item_id"]), str(row["sample_id"])))

    print(f"Output file now has {len(seen)} rows: {outputs_path}")


if __name__ == "__main__":
    main()
