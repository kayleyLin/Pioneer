"""Generate repeated outputs for formal RQ1 perturbed prompts.

Before running:

    export OPENAI_API_KEY="your_api_key_here"
    /Users/wenfenglin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/11_generate_rq1b_perturbed_outputs_openai.py

This script reads prompts/rq1_formal_perturbed_prompts.csv and writes
outputs/rq1_formal_perturbed_generations.csv by default.

To run a revised perturbation file:

    RQ1B_PERTURBED_PROMPTS=prompts/rq1_formal_perturbed_prompts.csv \
    RQ1B_PERTURBED_OUTPUTS=outputs/rq1_formal_perturbed_generations.csv \
    python3 src/11_generate_rq1b_perturbed_outputs_openai.py

To override the number of repeated generations for an exploratory run:

    RQ1B_N_SAMPLES=3 \
    RQ1B_PERTURBED_PROMPTS=prompts/rq1_surface_noise_intensity_prompts.csv \
    RQ1B_PERTURBED_OUTPUTS=outputs/rq1_surface_noise_intensity_generations.csv \
    python3 src/11_generate_rq1b_perturbed_outputs_openai.py
"""

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
DEFAULT_PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts.csv"
DEFAULT_OUTPUTS = ROOT / "outputs" / "rq1_formal_perturbed_generations.csv"
API_URL = "https://api.openai.com/v1/responses"
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


def make_ssl_context() -> ssl.SSLContext | None:
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run:\n"
            'export OPENAI_API_KEY="your_api_key_here"\n'
            "then run this script again."
        )

    config = read_config(CONFIG)
    prompts_path = env_path("RQ1B_PERTURBED_PROMPTS", DEFAULT_PROMPTS)
    outputs_path = env_path("RQ1B_PERTURBED_OUTPUTS", DEFAULT_OUTPUTS)
    prompts = read_csv(prompts_path)
    n_samples = env_int("RQ1B_N_SAMPLES", int(config["n_samples_per_prompt"]))

    total = len(prompts) * n_samples
    seen = existing_keys(outputs_path)
    if seen:
        print(f"Found {len(seen)} existing rows in {outputs_path}; resuming.")
    completed = 0

    for prompt_row in prompts:
        for sample_id in range(1, n_samples + 1):
            completed += 1
            key = (
                prompt_row["item_id"],
                prompt_row["perturbation_type"],
                str(sample_id),
            )
            if key in seen:
                print(
                    f"[{completed}/{total}] {prompt_row['item_id']} "
                    f"{prompt_row['perturbation_type']} sample {sample_id} skipped"
                )
                continue

            print(
                f"[{completed}/{total}] {prompt_row['item_id']} "
                f"{prompt_row['perturbation_type']} sample {sample_id}"
            )
            output_text = call_openai(prompt_row["perturbed_prompt"], config, api_key)
            append_row(
                outputs_path,
                {
                    "item_id": prompt_row["item_id"],
                    "task_type": prompt_row["task_type"],
                    "dataset_name": prompt_row["dataset_name"],
                    "source_index": prompt_row["source_index"],
                    "perturbation_type": prompt_row["perturbation_type"],
                    "sample_id": sample_id,
                    "model_name": config["model"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_output_tokens": config.get("max_output_tokens", ""),
                    "original_prompt": prompt_row["original_prompt"],
                    "perturbed_prompt": prompt_row["perturbed_prompt"],
                    "output_text": output_text,
                },
            )
            seen.add(key)
            time.sleep(0.2)

    print(f"Output file now has {len(seen)} rows: {outputs_path}")


if __name__ == "__main__":
    main()
