"""Generate repeated RQ1 outputs with the OpenAI Responses API.

Before running:

    export OPENAI_API_KEY="your_api_key_here"
    python3 src/07_generate_rq1_outputs_openai.py

This script reads prompts/rq1_sampled_original_prompts.csv and writes
outputs/rq1_formal_original_generations.csv by default.

To run a larger sample without overwriting the smaller formal pilot:

    python3 src/07_generate_rq1_outputs_openai.py \
        --input prompts/rq1_sampled_original_prompts_n50.csv \
        --output outputs/rq1_formal_original_generations_n50.csv
"""

import argparse
import csv
import importlib.util
import json
import os
import ssl
import time
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts.csv"
OUTPUTS = ROOT / "outputs" / "rq1_formal_original_generations.csv"
API_URL = "https://api.openai.com/v1/responses"
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
    """Use certifi certificates when available, which helps Anaconda Python."""
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
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def extract_output_text(response_data: dict) -> str:
    texts: list[str] = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip()


def call_openai(prompt: str, config: dict, api_key: str) -> str:
    payload = {
        "model": config["model"],
        "input": [
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
                raise RuntimeError(f"OpenAI API error {error.code}: {details}") from error
            wait_seconds = 5 * attempt
            print(
                f"OpenAI API error {error.code}; retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        except (RemoteDisconnected, TimeoutError, URLError) as error:
            if attempt == 5:
                raise RuntimeError(
                    f"OpenAI connection error after {attempt} attempts: {error}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROMPTS)
    parser.add_argument("--output", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run:\n"
            'export OPENAI_API_KEY="your_api_key_here"\n'
            "then run this script again."
        )

    config = read_config(CONFIG)
    prompts_path = resolve_path(args.input)
    outputs_path = resolve_path(args.output)
    prompts = read_csv(prompts_path)
    n_samples = int(config["n_samples_per_prompt"])

    total = len(prompts) * n_samples
    seen = existing_keys(outputs_path)
    if seen:
        print(f"Found {len(seen)} existing rows in {outputs_path}; resuming.")
    completed = 0

    for prompt_row in prompts:
        for sample_id in range(1, n_samples + 1):
            completed += 1
            key = (prompt_row["item_id"], str(sample_id))
            if key in seen:
                print(
                    f"[{completed}/{total}] "
                    f"{prompt_row['item_id']} sample {sample_id} skipped"
                )
                continue

            print(
                f"[{completed}/{total}] "
                f"{prompt_row['item_id']} sample {sample_id}"
            )
            output_text = call_openai(prompt_row["prompt_text"], config, api_key)
            append_row(
                outputs_path,
                {
                    "item_id": prompt_row["item_id"],
                    "task_type": prompt_row["task_type"],
                    "dataset_name": prompt_row["dataset_name"],
                    "source_index": prompt_row["source_index"],
                    "sample_id": sample_id,
                    "model_name": config["model"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_output_tokens": config.get("max_output_tokens", ""),
                    "prompt_text": prompt_row["prompt_text"],
                    "output_text": output_text,
                },
            )
            seen.add(key)
            time.sleep(0.2)

    print(f"Output file now has {len(seen)} rows: {outputs_path}")


if __name__ == "__main__":
    main()
