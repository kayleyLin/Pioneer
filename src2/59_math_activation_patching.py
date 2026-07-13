"""Activation patching probe for math paraphrasing drift on a cloud GPU.

Local use:
    python src/59_math_activation_patching.py --prepare-only

Cloud GPU use:
    python src/59_math_activation_patching.py --model meta-llama/Meta-Llama-3-8B-Instruct

The implemented intervention is a next-token recovery probe. For each case, the
target token is the first generated token under the original prompt. The script
measures whether replacing paraphrase activations at a given layer/token category
with the original category-mean activation recovers that target token logprob.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOUD_INPUTS = ROOT / "outputs" / "math_internal_cloud_probe_inputs.jsonl"
LOGPROB_INPUT = ROOT / "outputs" / "math_internal_logprob_probe_by_item.csv"
TRAJECTORY_INPUT = ROOT / "outputs" / "math_internal_trajectory_comparison.csv"
PATCH_CASES = ROOT / "outputs" / "math_internal_activation_patching_cases.jsonl"
PATCH_OUT = ROOT / "outputs" / "math_internal_activation_patching.csv"

NUMBER_RE = re.compile(r"\d")
MATH_RE = re.compile(r"(=|\+|-|\*|/|\^|<|>|\\|frac|sqrt|pi|theta|alpha|beta|gamma|sum|prod|log|sin|cos|tan)", re.I)
CONDITION_RE = re.compile(
    r"\b(at least|at most|exactly|remaining|difference|total|altogether|less than|greater than|if|when|given)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_cases(max_high: int, max_control: int) -> list[dict]:
    cloud = {row["item_id"]: row for row in read_jsonl(CLOUD_INPUTS)}
    logprob = {row["item_id"]: row for row in read_csv(LOGPROB_INPUT)}
    trajectory = {row["item_id"]: row for row in read_csv(TRAJECTORY_INPUT)}

    rows = []
    for item_id, record in cloud.items():
        lp = logprob[item_id]
        tr = trajectory[item_id]
        rows.append(
            {
                **record,
                "original_generated_text": lp["original_generated_text"],
                "paraphrase_generated_text": lp["paraphrase_generated_text"],
                "generated_mean_logprob_delta": float(lp["generated_mean_logprob_delta"]),
                "output_token_f1": float(tr["output_token_f1"]),
                "operation_match": tr["operation_match"],
                "final_answer_match": tr["final_answer_match"],
            }
        )

    high = sorted(
        [row for row in rows if "cross_model_shared_high" in row["selection_group"]],
        key=lambda row: (row["output_token_f1"], -row["mean_ncp"]),
    )[:max_high]
    controls = sorted(
        [row for row in rows if "low10" in row["selection_group"] and "cross_model_shared_high" not in row["selection_group"]],
        key=lambda row: (-row["output_token_f1"], abs(row["generated_mean_logprob_delta"])),
    )[:max_control]
    selected = []
    for row in high:
        row = dict(row)
        row["patch_group"] = "shared_high_drift"
        selected.append(row)
    for row in controls:
        row = dict(row)
        row["patch_group"] = "low_drift_control"
        selected.append(row)

    PATCH_CASES.parent.mkdir(parents=True, exist_ok=True)
    with PATCH_CASES.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return selected


def token_category(token: str) -> str:
    cleaned = token.replace("Ġ", " ").replace("▁", " ").strip()
    if NUMBER_RE.search(cleaned):
        return "number"
    if MATH_RE.search(cleaned):
        return "math"
    if CONDITION_RE.search(cleaned):
        return "condition"
    if cleaned and cleaned.isalpha():
        return "content"
    return "other"


def category_indices(categories: list[str], category: str) -> list[int]:
    if category == "all":
        return list(range(len(categories)))
    return [index for index, value in enumerate(categories) if value == category]


def first_token_id(tokenizer, text: str) -> int | None:
    encoded = tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
    return encoded[0] if encoded else None


def target_logprob(model, encoded, target_token_id: int) -> float:
    import torch

    with torch.no_grad():
        outputs = model(**encoded, use_cache=False, return_dict=True)
        logits = outputs.logits[0, -1, :]
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    return logprobs[target_token_id].item()


def run_patching(
    *,
    cases_path: Path,
    output_path: Path,
    model_name: str,
    device_map: str,
    torch_dtype: str,
    max_cases: int | None,
    categories: list[str],
    layer_stride: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    cases = read_jsonl(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]

    rows = []
    for case in cases:
        original_encoded = tokenizer(case["original_model_prompt"], return_tensors="pt").to(model.device)
        paraphrase_encoded = tokenizer(case["paraphrase_model_prompt"], return_tensors="pt").to(model.device)
        target_id = first_token_id(tokenizer, case["original_generated_text"])
        if target_id is None:
            continue

        original_tokens = tokenizer.convert_ids_to_tokens(original_encoded["input_ids"][0].tolist())
        paraphrase_tokens = tokenizer.convert_ids_to_tokens(paraphrase_encoded["input_ids"][0].tolist())
        original_categories = [token_category(token) for token in original_tokens]
        paraphrase_categories = [token_category(token) for token in paraphrase_tokens]

        original_target_logprob = target_logprob(model, original_encoded, target_id)
        paraphrase_target_logprob = target_logprob(model, paraphrase_encoded, target_id)

        with torch.no_grad():
            original_outputs = model(
                **original_encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        n_layers = len(model.model.layers)
        for layer in range(0, n_layers, layer_stride):
            original_layer_hidden = original_outputs.hidden_states[layer + 1][0]
            for category in categories:
                orig_indices = category_indices(original_categories, category)
                para_indices = category_indices(paraphrase_categories, category)
                if not orig_indices or not para_indices:
                    continue
                original_mean = original_layer_hidden[
                    torch.tensor(orig_indices, device=original_layer_hidden.device)
                ].mean(dim=0)

                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    patched = hidden.clone()
                    patch_indices = torch.tensor(para_indices, device=patched.device)
                    patched[0, patch_indices, :] = original_mean.to(patched.device, patched.dtype)
                    if isinstance(output, tuple):
                        return (patched, *output[1:])
                    return patched

                handle = model.model.layers[layer].register_forward_hook(hook)
                try:
                    patched_logprob = target_logprob(model, paraphrase_encoded, target_id)
                finally:
                    handle.remove()

                rows.append(
                    {
                        "item_id": case["item_id"],
                        "patch_group": case["patch_group"],
                        "selection_group": case["selection_group"],
                        "layer": layer,
                        "token_category": category,
                        "target_token_id": target_id,
                        "original_target_logprob": original_target_logprob,
                        "paraphrase_target_logprob": paraphrase_target_logprob,
                        "patched_target_logprob": patched_logprob,
                        "recovery_over_paraphrase": patched_logprob - paraphrase_target_logprob,
                        "remaining_gap_to_original": original_target_logprob - patched_logprob,
                    }
                )

    write_csv(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-high", type=int, default=5)
    parser.add_argument("--max-control", type=int, default=5)
    parser.add_argument("--cases-path", type=Path, default=PATCH_CASES)
    parser.add_argument("--output", type=Path, default=PATCH_OUT)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--categories", default="number,math,condition,content,all")
    parser.add_argument("--layer-stride", type=int, default=2)
    args = parser.parse_args()

    selected = prepare_cases(args.max_high, args.max_control)
    print(f"wrote {PATCH_CASES} records={len(selected)}")
    if args.prepare_only:
        return
    run_patching(
        cases_path=args.cases_path,
        output_path=args.output,
        model_name=args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_cases=args.max_cases,
        categories=[item.strip() for item in args.categories.split(",") if item.strip()],
        layer_stride=args.layer_stride,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
