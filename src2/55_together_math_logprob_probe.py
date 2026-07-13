"""Probe math paraphrasing drift with Together completion logprobs.

This script uses Together's OpenAI-compatible completions endpoint because it can
return prompt logprobs with echo=true. It compares:

1. prompt likelihood for original vs fixed paraphrased prompts;
2. likelihood of the same reference continuation under each prompt condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
INPUT = ROOT / "outputs" / "math_internal_case_selection.csv"
RAW_OUT = ROOT / "outputs" / "math_internal_logprob_probe_raw.jsonl"
BY_ITEM_OUT = ROOT / "outputs" / "math_internal_logprob_probe_by_item.csv"
SUMMARY_OUT = ROOT / "outputs" / "math_internal_logprob_probe_summary.csv"
MD_OUT = ROOT / "outputs" / "math_internal_logprob_probe_summary.md"

DEFAULT_MODEL = os.environ.get("TOGETHER_LOGPROB_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct-Lite")
DEFAULT_API_URL = os.environ.get("TOGETHER_COMPLETIONS_URL", "https://api.together.ai/v1/completions")

MATH_TOKEN_RE = re.compile(
    r"(\d|[=+\-*/^<>]|\\|frac|sqrt|pi|theta|alpha|beta|gamma|sum|prod|log|sin|cos|tan)",
    re.IGNORECASE,
)


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def llama3_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def finite_mean(values: list[float | None]) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def math_token_mean(tokens: list[str], logprobs: list[float | None]) -> float:
    selected = [
        logprob
        for token, logprob in zip(tokens, logprobs)
        if logprob is not None and MATH_TOKEN_RE.search(token)
    ]
    return finite_mean(selected)


def make_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def call_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    echo: bool = True,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "logprobs": 1,
        "echo": echo,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "pioneer-math-logprob-probe/1.0",
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    for attempt in range(1, 6):
        try:
            with urlopen(request, timeout=timeout, context=make_ssl_context()) as response:
                return json.load(response)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 5:
                raise RuntimeError(f"Together completion error {error.code}: {details}") from error
            time.sleep(5 * attempt)
        except (TimeoutError, URLError) as error:
            if attempt == 5:
                raise RuntimeError(f"Together connection error after {attempt} attempts: {error}") from error
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def extract_logprobs(response: dict) -> tuple[list[str], list[float | None]]:
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")

    if tokens is not None and token_logprobs is not None:
        return tokens, token_logprobs

    # Some OpenAI-compatible APIs use content-style logprob blocks.
    content = logprobs.get("content")
    if content:
        parsed_tokens = []
        parsed_logprobs = []
        for item in content:
            parsed_tokens.append(item.get("token", ""))
            parsed_logprobs.append(item.get("logprob"))
        return parsed_tokens, parsed_logprobs

    raise ValueError(f"Unsupported logprobs response shape: {json.dumps(choice)[:1000]}")


def probe_prompt(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
) -> dict[str, object]:
    response = call_completion(
        api_url=api_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_tokens=1,
        timeout=timeout,
    )
    tokens, token_logprobs = extract_logprobs(response)
    return {
        "token_count": len(tokens),
        "mean_logprob": finite_mean(token_logprobs),
        "math_token_mean_logprob": math_token_mean(tokens, token_logprobs),
        "tokens": tokens,
        "token_logprobs": token_logprobs,
    }


def probe_generation(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, object]:
    response = call_completion(
        api_url=api_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        echo=False,
    )
    tokens, token_logprobs = extract_logprobs(response)
    choice = response["choices"][0]
    text = choice.get("text") or choice.get("message", {}).get("content", "")
    return {
        "generated_text": str(text).strip(),
        "generated_token_count": len(tokens),
        "generated_mean_logprob": finite_mean(token_logprobs),
        "generated_math_token_mean_logprob": math_token_mean(tokens, token_logprobs),
        "generated_tokens": tokens,
        "generated_token_logprobs": token_logprobs,
    }


def continuation_logprob(
    *,
    api_url: str,
    api_key: str,
    model: str,
    base_prompt: str,
    reference: str,
    timeout: int,
) -> dict[str, object]:
    base = probe_prompt(api_url=api_url, api_key=api_key, model=model, prompt=base_prompt, timeout=timeout)
    full = probe_prompt(
        api_url=api_url,
        api_key=api_key,
        model=model,
        prompt=base_prompt + reference,
        timeout=timeout,
    )

    base_count = int(base["token_count"])
    full_tokens = full["tokens"]
    full_logprobs = full["token_logprobs"]
    suffix_tokens = full_tokens[base_count:]
    suffix_logprobs = full_logprobs[base_count:]
    if not suffix_tokens and len(full_tokens) > len(base["tokens"]):
        suffix_tokens = full_tokens[len(base["tokens"]) :]
        suffix_logprobs = full_logprobs[len(base["tokens"]) :]

    return {
        "base_token_count": base_count,
        "full_token_count": int(full["token_count"]),
        "reference_token_count": len(suffix_tokens),
        "reference_mean_logprob": finite_mean(suffix_logprobs),
        "reference_math_token_mean_logprob": math_token_mean(suffix_tokens, suffix_logprobs),
        "base_prompt_mean_logprob": base["mean_logprob"],
        "base_prompt_math_token_mean_logprob": base["math_token_mean_logprob"],
        "reference_tokens": suffix_tokens,
        "reference_token_logprobs": suffix_logprobs,
    }


def select_cases(rows: list[dict[str, str]], max_cases: int | None, shared_only: bool) -> list[dict[str, str]]:
    if shared_only:
        rows = [row for row in rows if "cross_model_shared_high" in row["selection_group"]]
    rows = sorted(
        rows,
        key=lambda row: (
            "cross_model_shared_high" not in row["selection_group"],
            float(row["mean_rank"]),
            -float(row["mean_ncp"]),
        ),
    )
    if max_cases is not None:
        rows = rows[:max_cases]
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    keys = [
        "prompt_mean_logprob_delta",
        "prompt_math_logprob_delta",
        "reference_mean_logprob_delta",
        "reference_math_logprob_delta",
        "generated_mean_logprob_delta",
        "generated_math_logprob_delta",
    ]
    summary = []
    for group_name, group_rows in [
        ("all", rows),
        ("shared_high", [row for row in rows if "cross_model_shared_high" in str(row["selection_group"])]),
        ("non_shared", [row for row in rows if "cross_model_shared_high" not in str(row["selection_group"])]),
    ]:
        if not group_rows:
            continue
        item: dict[str, object] = {"group": group_name, "n_items": len(group_rows)}
        for key in keys:
            values = [float(row[key]) for row in group_rows if key in row and row[key] == row[key]]
            item[f"mean_{key}"] = sum(values) / len(values) if values else math.nan
            item[f"negative_{key}_count"] = sum(value < 0 for value in values)
        summary.append(item)
    return summary


def write_md(by_item_rows: list[dict[str, object]], summary_rows: list[dict[str, object]], model: str) -> None:
    lines = [
        "# Math internal logprob probe",
        "",
        f"- Model: `{model}`",
        f"- Items probed: {len(by_item_rows)}",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- {group}: n={n_items}, mean reference logprob delta={ref:.6f}, "
            "negative reference delta count={neg_ref}, mean generated logprob delta={gen:.6f}".format(
                group=row["group"],
                n_items=row["n_items"],
                ref=row["mean_reference_mean_logprob_delta"],
                neg_ref=row["negative_reference_mean_logprob_delta_count"],
                gen=row["mean_generated_mean_logprob_delta"],
            )
        )
    lines.extend(
        [
            "",
            "Definition:",
            "",
            "`delta = paraphrased_prompt_condition - original_prompt_condition`.",
            "A negative reference delta means the original Llama continuation is less likely under the paraphrased prompt.",
            "If reference deltas are NaN, the hosted endpoint did not expose prompt echo logprobs for this model; generated-token logprob deltas are then the usable API-only signal.",
            "",
            "## Strongest negative reference-likelihood shifts",
            "",
        ]
    )
    reference_rows = [
        row for row in by_item_rows if row.get("reference_mean_logprob_delta") == row.get("reference_mean_logprob_delta")
    ]
    for row in sorted(reference_rows, key=lambda item: float(item["reference_mean_logprob_delta"]))[:10]:
        lines.append(
            f"- {row['item_id']}: reference_delta={float(row['reference_mean_logprob_delta']):.6f}, "
            f"prompt_delta={float(row['prompt_mean_logprob_delta']):.6f}, groups={row['selection_group']}"
        )
    if not reference_rows:
        lines.append("- Not available: Together did not return prompt echo logprobs for fixed-reference scoring.")
    lines.extend(["", "## Strongest negative generated-token confidence shifts", ""])
    generated_rows = [
        row for row in by_item_rows if row.get("generated_mean_logprob_delta") == row.get("generated_mean_logprob_delta")
    ]
    for row in sorted(generated_rows, key=lambda item: float(item["generated_mean_logprob_delta"]))[:10]:
        lines.append(
            f"- {row['item_id']}: generated_delta={float(row['generated_mean_logprob_delta']):.6f}, "
            f"groups={row['selection_group']}"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUT)
    parser.add_argument("--by-item-output", type=Path, default=BY_ITEM_OUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--reference-chars", type=int, default=1200)
    parser.add_argument("--generated-max-tokens", type=int, default=220)
    parser.add_argument("--generated-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set TOGETHER_API_KEY before running this script.")

    config = read_config()
    cases = select_cases(read_cases(args.input), args.max_cases, args.shared_only)
    raw_path = args.raw_output
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    by_item_rows: list[dict[str, object]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, row in enumerate(cases, start=1):
            item_id = row["item_id"]
            reference = row["llama_original_output_sample1"][: args.reference_chars].strip()
            if not reference:
                raise ValueError(f"{item_id} has empty Llama original output sample")

            original_base = llama3_prompt(config["system_prompt"], row["original_prompt"])
            paraphrase_base = llama3_prompt(config["system_prompt"], row["perturbed_prompt"])

            if args.generated_only:
                original = {
                    "base_prompt_mean_logprob": math.nan,
                    "base_prompt_math_token_mean_logprob": math.nan,
                    "reference_mean_logprob": math.nan,
                    "reference_math_token_mean_logprob": math.nan,
                    "reference_token_count": 0,
                }
                paraphrase = dict(original)
            else:
                original = continuation_logprob(
                    api_url=args.api_url,
                    api_key=api_key,
                    model=args.model,
                    base_prompt=original_base,
                    reference=reference,
                    timeout=args.timeout,
                )
                paraphrase = continuation_logprob(
                    api_url=args.api_url,
                    api_key=api_key,
                    model=args.model,
                    base_prompt=paraphrase_base,
                    reference=reference,
                    timeout=args.timeout,
                )
            original_generated = probe_generation(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                prompt=original_base,
                max_tokens=args.generated_max_tokens,
                timeout=args.timeout,
            )
            paraphrase_generated = probe_generation(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                prompt=paraphrase_base,
                max_tokens=args.generated_max_tokens,
                timeout=args.timeout,
            )

            raw_file.write(
                json.dumps(
                    {
                        "item_id": item_id,
                        "selection_group": row["selection_group"],
                        "reference_chars": len(reference),
                        "original": original,
                        "paraphrase": paraphrase,
                        "original_generated": original_generated,
                        "paraphrase_generated": paraphrase_generated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_file.flush()

            result = {
                "item_id": item_id,
                "selection_group": row["selection_group"],
                "mean_ncp": row["mean_ncp"],
                "mean_rank": row["mean_rank"],
                "reference_chars": len(reference),
                "original_prompt_mean_logprob": original["base_prompt_mean_logprob"],
                "paraphrase_prompt_mean_logprob": paraphrase["base_prompt_mean_logprob"],
                "prompt_mean_logprob_delta": paraphrase["base_prompt_mean_logprob"]
                - original["base_prompt_mean_logprob"],
                "original_prompt_math_logprob": original["base_prompt_math_token_mean_logprob"],
                "paraphrase_prompt_math_logprob": paraphrase["base_prompt_math_token_mean_logprob"],
                "prompt_math_logprob_delta": paraphrase["base_prompt_math_token_mean_logprob"]
                - original["base_prompt_math_token_mean_logprob"],
                "original_reference_mean_logprob": original["reference_mean_logprob"],
                "paraphrase_reference_mean_logprob": paraphrase["reference_mean_logprob"],
                "reference_mean_logprob_delta": paraphrase["reference_mean_logprob"]
                - original["reference_mean_logprob"],
                "original_reference_math_logprob": original["reference_math_token_mean_logprob"],
                "paraphrase_reference_math_logprob": paraphrase["reference_math_token_mean_logprob"],
                "reference_math_logprob_delta": paraphrase["reference_math_token_mean_logprob"]
                - original["reference_math_token_mean_logprob"],
                "original_reference_token_count": original["reference_token_count"],
                "paraphrase_reference_token_count": paraphrase["reference_token_count"],
                "original_generated_mean_logprob": original_generated["generated_mean_logprob"],
                "paraphrase_generated_mean_logprob": paraphrase_generated["generated_mean_logprob"],
                "generated_mean_logprob_delta": paraphrase_generated["generated_mean_logprob"]
                - original_generated["generated_mean_logprob"],
                "original_generated_math_logprob": original_generated["generated_math_token_mean_logprob"],
                "paraphrase_generated_math_logprob": paraphrase_generated["generated_math_token_mean_logprob"],
                "generated_math_logprob_delta": paraphrase_generated["generated_math_token_mean_logprob"]
                - original_generated["generated_math_token_mean_logprob"],
                "original_generated_token_count": original_generated["generated_token_count"],
                "paraphrase_generated_token_count": paraphrase_generated["generated_token_count"],
                "original_generated_text": original_generated["generated_text"],
                "paraphrase_generated_text": paraphrase_generated["generated_text"],
            }
            by_item_rows.append(result)
            print(
                f"[{index}/{len(cases)}] {item_id} "
                f"reference_delta={result['reference_mean_logprob_delta']:.6f} "
                f"generated_delta={result['generated_mean_logprob_delta']:.6f}",
                flush=True,
            )

    summary_rows = summarize(by_item_rows)
    write_csv(args.by_item_output, by_item_rows)
    write_csv(args.summary_output, summary_rows)
    write_md(by_item_rows, summary_rows, args.model)
    print(f"wrote {args.raw_output}")
    print(f"wrote {args.by_item_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
