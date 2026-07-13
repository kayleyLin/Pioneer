"""Cloud-GPU hidden-state and attention probe for math paraphrasing.

Run locally with --prepare-only to create the JSONL input package. Run on a GPU
machine with transformers/torch installed to collect hidden-state and attention
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
CASE_SELECTION = ROOT / "outputs" / "math_internal_case_selection.csv"
INPUT_JSONL = ROOT / "outputs" / "math_internal_cloud_probe_inputs.jsonl"
HIDDEN_OUT = ROOT / "outputs" / "math_internal_hidden_state_by_layer.csv"
ATTENTION_OUT = ROOT / "outputs" / "math_internal_attention_by_token_type.csv"

NUMBER_RE = re.compile(r"\d")
MATH_RE = re.compile(r"(=|\+|-|\*|/|\^|<|>|\\|frac|sqrt|pi|theta|alpha|beta|gamma|sum|prod|log|sin|cos|tan)", re.I)
CONDITION_RE = re.compile(
    r"\b(at least|at most|exactly|remaining|difference|total|altogether|less than|greater than|if|when|given)\b",
    re.I,
)


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def llama3_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def prepare_inputs(input_csv: Path, output_jsonl: Path, max_cases: int | None, shared_only: bool) -> list[dict]:
    config = read_config()
    with input_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
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

    records = []
    for row in rows:
        records.append(
            {
                "item_id": row["item_id"],
                "selection_group": row["selection_group"],
                "mean_ncp": float(row["mean_ncp"]),
                "mean_rank": float(row["mean_rank"]),
                "original_user_prompt": row["original_prompt"],
                "paraphrase_user_prompt": row["perturbed_prompt"],
                "original_model_prompt": llama3_prompt(config["system_prompt"], row["original_prompt"]),
                "paraphrase_model_prompt": llama3_prompt(config["system_prompt"], row["perturbed_prompt"]),
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


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


def cosine(left, right) -> float:
    import torch

    if left is None or right is None:
        return math.nan
    return torch.nn.functional.cosine_similarity(left, right, dim=0).item()


def category_mean(hidden, categories: list[str], category: str):
    import torch

    indices = [index for index, value in enumerate(categories) if category == "all" or value == category]
    if not indices:
        return None
    index_tensor = torch.tensor(indices, device=hidden.device)
    return hidden.index_select(0, index_tensor).mean(dim=0)


def run_model_probe(
    *,
    input_jsonl: Path,
    model_name: str,
    hidden_output: Path,
    attention_output: Path,
    device_map: str,
    torch_dtype: str,
    max_cases: int | None,
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

    with input_jsonl.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    if max_cases is not None:
        records = records[:max_cases]

    hidden_rows = []
    attention_rows = []
    categories_to_probe = ["all", "number", "math", "condition", "content"]

    for record in records:
        encoded = {}
        outputs = {}
        categories = {}
        for condition, prompt_key in [
            ("original", "original_model_prompt"),
            ("paraphrase", "paraphrase_model_prompt"),
        ]:
            encoded[condition] = tokenizer(record[prompt_key], return_tensors="pt").to(model.device)
            token_ids = encoded[condition]["input_ids"][0].tolist()
            token_texts = tokenizer.convert_ids_to_tokens(token_ids)
            categories[condition] = [token_category(token) for token in token_texts]
            with torch.no_grad():
                outputs[condition] = model(
                    **encoded[condition],
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )

        for layer_index, (orig_hidden, para_hidden) in enumerate(
            zip(outputs["original"].hidden_states, outputs["paraphrase"].hidden_states)
        ):
            orig_hidden = orig_hidden[0]
            para_hidden = para_hidden[0]
            for category in categories_to_probe:
                hidden_rows.append(
                    {
                        "item_id": record["item_id"],
                        "selection_group": record["selection_group"],
                        "layer": layer_index,
                        "token_category": category,
                        "cosine_similarity": cosine(
                            category_mean(orig_hidden, categories["original"], category),
                            category_mean(para_hidden, categories["paraphrase"], category),
                        ),
                    }
                )

        for condition in ["original", "paraphrase"]:
            cat = categories[condition]
            for layer_index, attention in enumerate(outputs[condition].attentions):
                final_token_attention = attention[0, :, -1, :].mean(dim=0)
                for category in categories_to_probe:
                    indices = [index for index, value in enumerate(cat) if category == "all" or value == category]
                    mass = math.nan
                    if indices:
                        mass = final_token_attention[torch.tensor(indices, device=final_token_attention.device)].sum().item()
                    attention_rows.append(
                        {
                            "item_id": record["item_id"],
                            "selection_group": record["selection_group"],
                            "condition": condition,
                            "layer": layer_index,
                            "token_category": category,
                            "final_token_attention_mass": mass,
                            "token_count": len(indices),
                        }
                    )

    write_rows(hidden_output, hidden_rows)
    write_rows(attention_output, attention_rows)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-selection", type=Path, default=CASE_SELECTION)
    parser.add_argument("--input-jsonl", type=Path, default=INPUT_JSONL)
    parser.add_argument("--hidden-output", type=Path, default=HIDDEN_OUT)
    parser.add_argument("--attention-output", type=Path, default=ATTENTION_OUT)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    records = prepare_inputs(args.case_selection, args.input_jsonl, args.max_cases, args.shared_only)
    print(f"wrote {args.input_jsonl} records={len(records)}")
    if args.prepare_only:
        return
    run_model_probe(
        input_jsonl=args.input_jsonl,
        model_name=args.model,
        hidden_output=args.hidden_output,
        attention_output=args.attention_output,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_cases=args.max_cases,
    )
    print(f"wrote {args.hidden_output}")
    print(f"wrote {args.attention_output}")


if __name__ == "__main__":
    main()
