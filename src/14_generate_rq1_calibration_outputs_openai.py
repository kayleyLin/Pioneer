"""Generate 10 repeated outputs for each RQ1 calibration prompt."""

import csv
import importlib.util
import json
import os
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
PROMPTS = ROOT / "prompts" / "rq1_calibration_prompts.csv"
OUTPUTS = ROOT / "outputs" / "rq1_calibration_generations.csv"
API_URL = "https://api.openai.com/v1/responses"
N_SAMPLES = 10


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

    try:
        with urlopen(request, timeout=120, context=make_ssl_context()) as response:
            response_data = json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {error.code}: {details}") from error

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
    prompts = read_csv(PROMPTS)

    rows: list[dict[str, str | int | float]] = []
    total = len(prompts) * N_SAMPLES
    completed = 0

    for prompt_row in prompts:
        for sample_id in range(1, N_SAMPLES + 1):
            completed += 1
            print(f"[{completed}/{total}] {prompt_row['item_id']} sample {sample_id}")
            output_text = call_openai(prompt_row["prompt_text"], config, api_key)
            rows.append(
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
                }
            )
            time.sleep(0.2)

    with OUTPUTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUTS}")


if __name__ == "__main__":
    main()
