"""Generate RQ1 temperature-pilot outputs with the OpenAI Responses API.

This script uses the existing RQ1 calibration prompt set and generates repeated
outputs across two temperature sweeps:

    coarse: 0.0, 0.3, 0.7, 1.0
    local:  0.5, 0.7, 0.8, 0.9

By default, the union is generated:

    0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0

To run in two stages:

    TEMPERATURE_PILOT_PHASE=coarse python3 src/17_generate_rq1_temperature_pilot_openai.py
    TEMPERATURE_PILOT_PHASE=local python3 src/17_generate_rq1_temperature_pilot_openai.py

Existing temperature=0.7 calibration outputs are reused when available, so the
script only needs to call the API for the new temperature values. The script is
resumable: if outputs/rq1_temperature_pilot_generations.csv already contains
rows for a prompt/temperature/sample_id combination, that call is skipped.
"""

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
EXISTING_CALIBRATION = ROOT / "outputs" / "rq1_calibration_generations.csv"
OUTPUTS = ROOT / "outputs" / "rq1_temperature_pilot_generations.csv"
API_URL = "https://api.openai.com/v1/responses"

N_SAMPLES = 5
COARSE_TEMPERATURES = {0.0, 0.3, 0.7, 1.0}
LOCAL_TEMPERATURES = {0.5, 0.7, 0.8, 0.9}
ALL_TEMPERATURES = sorted(COARSE_TEMPERATURES | LOCAL_TEMPERATURES)

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
    "temperature_phase",
    "generation_source",
    "prompt_text",
    "output_text",
]


def make_ssl_context() -> ssl.SSLContext | None:
    if importlib.util.find_spec("certifi") is None:
        return None

    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(rows: list[dict[str, str | int | float]]) -> None:
    OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def temperature_phase(temperature: float) -> str:
    in_coarse = temperature in COARSE_TEMPERATURES
    in_local = temperature in LOCAL_TEMPERATURES
    if in_coarse and in_local:
        return "coarse_and_local"
    if in_coarse:
        return "coarse"
    if in_local:
        return "local"
    return "other"


def selected_temperatures() -> list[float]:
    phase = os.environ.get("TEMPERATURE_PILOT_PHASE", "full").strip().lower()
    if phase == "coarse":
        return sorted(COARSE_TEMPERATURES)
    if phase == "local":
        return sorted(LOCAL_TEMPERATURES)
    if phase in {"full", "all", "both"}:
        return ALL_TEMPERATURES
    raise SystemExit(
        "Invalid TEMPERATURE_PILOT_PHASE. Use one of: coarse, local, full."
    )


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["item_id"], str(row["temperature"]), str(row["sample_id"]))


def extract_output_text(response_data: dict) -> str:
    texts: list[str] = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip()


def call_openai(prompt: str, config: dict, api_key: str, temperature: float) -> str:
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
        "temperature": temperature,
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


def reusable_temperature_07_rows(config: dict) -> list[dict[str, str | int | float]]:
    rows = read_csv(EXISTING_CALIBRATION)
    reused: list[dict[str, str | int | float]] = []

    for row in rows:
        if float(row["temperature"]) != 0.7:
            continue
        if int(row["sample_id"]) > N_SAMPLES:
            continue
        reused.append(
            {
                "item_id": row["item_id"],
                "task_type": row["task_type"],
                "dataset_name": row["dataset_name"],
                "source_index": row["source_index"],
                "sample_id": int(row["sample_id"]),
                "model_name": row.get("model_name", config["model"]),
                "temperature": 0.7,
                "top_p": row.get("top_p", config["top_p"]),
                "max_output_tokens": row.get("max_output_tokens", ""),
                "temperature_phase": temperature_phase(0.7),
                "generation_source": "reused_rq1_calibration_generations",
                "prompt_text": row["prompt_text"],
                "output_text": row["output_text"],
            }
        )

    return reused


def main() -> None:
    config = read_config(CONFIG)
    prompts = read_csv(PROMPTS)
    existing_rows = read_csv(OUTPUTS)
    temperatures = selected_temperatures()

    rows: list[dict[str, str | int | float]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in existing_rows:
        rows.append(row)
        seen.add(row_key(row))

    for row in reusable_temperature_07_rows(config):
        key = (str(row["item_id"]), str(row["temperature"]), str(row["sample_id"]))
        if key not in seen:
            rows.append(row)
            seen.add(key)

    write_rows(rows)

    api_key = os.environ.get("OPENAI_API_KEY")
    required_calls = [
        (prompt_row, temperature, sample_id)
        for prompt_row in prompts
        for temperature in temperatures
        for sample_id in range(1, N_SAMPLES + 1)
        if (prompt_row["item_id"], str(temperature), str(sample_id)) not in seen
    ]

    if not required_calls:
        print(f"All temperature-pilot rows already exist in {OUTPUTS}")
        return

    if not api_key:
        raise SystemExit(
            f"Prepared reusable rows in {OUTPUTS}, but OPENAI_API_KEY is not set.\n"
            "Run:\n"
            'export OPENAI_API_KEY="your_api_key_here"\n'
            "then run this script again."
        )

    total = len(required_calls)
    print(f"Selected temperatures: {temperatures}")
    for completed, (prompt_row, temperature, sample_id) in enumerate(
        required_calls, start=1
    ):
        print(
            f"[{completed}/{total}] {prompt_row['item_id']} "
            f"temp={temperature} sample={sample_id}"
        )
        output_text = call_openai(
            prompt_row["prompt_text"], config, api_key, temperature
        )
        rows.append(
            {
                "item_id": prompt_row["item_id"],
                "task_type": prompt_row["task_type"],
                "dataset_name": prompt_row["dataset_name"],
                "source_index": prompt_row["source_index"],
                "sample_id": sample_id,
                "model_name": config["model"],
                "temperature": temperature,
                "top_p": config["top_p"],
                "max_output_tokens": config.get("max_output_tokens", ""),
                "temperature_phase": temperature_phase(temperature),
                "generation_source": "api_new",
                "prompt_text": prompt_row["prompt_text"],
                "output_text": output_text,
            }
        )
        write_rows(rows)
        time.sleep(0.2)

    print(f"Wrote {len(rows)} rows to {OUTPUTS}")


if __name__ == "__main__":
    main()
