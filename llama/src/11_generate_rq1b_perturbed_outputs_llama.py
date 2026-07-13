"""Generate repeated outputs for RQ1 perturbed prompts with an OpenAI-compatible Llama API.

Before running:

    export LLAMA_API_KEY="your_api_key_here"
    python3 src/11_generate_rq1b_perturbed_outputs_llama.py

This script reads prompts/rq1_formal_perturbed_prompts_n50_*.csv and writes
llama/outputs/rq1_llama_perturbed_generations_n50.csv by default.

Per-task override:

    RQ1B_LLAMA_PERTURBED_PROMPTS=prompts/rq1_formal_perturbed_prompts_n50_factual_qa.csv \
    RQ1B_LLAMA_PERTURBED_OUTPUTS=llama/outputs/rq1_llama_perturbed_generations_n50_factual_qa.csv \
    python3 src/11_generate_rq1b_perturbed_outputs_llama.py
"""

import csv
import importlib.util
import json
import os
import ssl
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
DEFAULT_PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_factual_qa.csv"
DEFAULT_OUTPUTS = ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa.csv"
API_URL = os.environ.get("LLAMA_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
FIELDNAMES = [
    "item_id",
    "task_type",
    "dataset_name",
    "source_index",
    "perturbation_type",
    "sample_id",
    "model_name",
    "temperature",
    "top_p",
    "max_output_tokens",
    "original_prompt",
    "perturbed_prompt",
    "output_text",
]


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer, got: {value}") from error
    if parsed < 1:
        raise SystemExit(f"{name} must be at least 1, got: {value}")
    return parsed


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def make_ssl_context() -> ssl.SSLContext | None:
    if API_URL.startswith("http://"):
        return None
    if importlib.util.find_spec("certifi") is None:
        return None
    import certifi
    return ssl.create_default_context(cafile=certifi.where())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    rows = read_csv(path)
    return {
        (row["item_id"], row["perturbation_type"], row["sample_id"])
        for row in rows
    }


def append_row(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def extract_output_text(response_data: dict) -> str:
    """Extract text from Llama chat/completions response."""
    return response_data["choices"][0]["message"]["content"].strip()


def call_llama(prompt: str, config: dict, api_key: str) -> str:
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

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "pioneer-llama-rq1/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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
                raise RuntimeError(f"Llama API error {error.code}: {details}") from error
            wait_seconds = 5 * attempt
            print(
                f"Llama API error {error.code}; retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        except (IncompleteRead, RemoteDisconnected, TimeoutError, URLError) as error:
            if attempt == 5:
                raise RuntimeError(
                    f"Llama connection error after {attempt} attempts: {error}"
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
        "perturbation_type": prompt_row["perturbation_type"],
        "sample_id": sample_id,
        "model_name": MODEL,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_output_tokens": config.get("max_output_tokens", ""),
        "original_prompt": prompt_row["original_prompt"],
        "perturbed_prompt": prompt_row["perturbed_prompt"],
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
        f"[{job_number}/{total}] {prompt_row['item_id']} "
        f"{prompt_row['perturbation_type']} sample {sample_id}",
        flush=True,
    )
    output_text = call_llama(prompt_row["perturbed_prompt"], config, api_key)
    return job_number, make_output_row(prompt_row, sample_id, output_text, config)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.samples is not None and args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.limit_rows is not None and args.limit_rows < 1:
        raise SystemExit("--limit-rows must be at least 1")

    api_key = os.environ.get("LLAMA_API_KEY", "")

    config = read_config(CONFIG)
    prompts_path = (
        resolve_path(args.input)
        if args.input is not None
        else env_path("RQ1B_LLAMA_PERTURBED_PROMPTS", DEFAULT_PROMPTS)
    )
    outputs_path = (
        resolve_path(args.output)
        if args.output is not None
        else env_path("RQ1B_LLAMA_PERTURBED_OUTPUTS", DEFAULT_OUTPUTS)
    )
    prompts = read_csv(prompts_path)
    if args.limit_rows is not None:
        prompts = prompts[: args.limit_rows]
    n_samples = args.samples or env_int(
        "RQ1B_LLAMA_N_SAMPLES", int(config["n_samples_per_prompt"])
    )

    total = len(prompts) * n_samples
    seen = existing_keys(outputs_path)
    if seen:
        print(f"Found {len(seen)} existing rows in {outputs_path}; resuming.")
    jobs: list[tuple[int, dict[str, str], int]] = []
    job_number = 0

    for prompt_row in prompts:
        for sample_id in range(1, n_samples + 1):
            job_number += 1
            key = (
                prompt_row["item_id"],
                prompt_row["perturbation_type"],
                str(sample_id),
            )
            if key in seen:
                print(
                    f"[{job_number}/{total}] {prompt_row['item_id']} "
                    f"{prompt_row['perturbation_type']} sample {sample_id} skipped"
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
            seen.add(
                (
                    str(row["item_id"]),
                    str(row["perturbation_type"]),
                    str(row["sample_id"]),
                )
            )
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
                seen.add(
                    (
                        str(row["item_id"]),
                        str(row["perturbation_type"]),
                        str(row["sample_id"]),
                    )
                )

    print(f"Output file now has {len(seen)} rows: {outputs_path}")


if __name__ == "__main__":
    main()




