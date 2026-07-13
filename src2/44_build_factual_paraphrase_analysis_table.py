"""Build item-level factual-QA paraphrasing analysis tables.

The perturbed factual-QA paraphrasing files in this repository store the actual
perturbed prompt as the paraphrased question only, without the original context.
This script preserves that actual prompt in `perturbed_prompt` and adds a
separate reconstructed prompt with the original context for later audit work.

Outputs:
    outputs/factual_paraphrase_item_table.csv
    qwen/outputs/factual_paraphrase_item_table.csv
    llama/outputs/factual_paraphrase_item_table.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50_factual_qa.csv"
DEFAULT_PERTURBED_PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_factual_qa.csv"

TASK = "factual_qa"
PERTURBATION = "paraphrasing"

BRANCHES = {
    "outputs": {
        "output_dir": ROOT / "outputs",
        "output_name": "factual_paraphrase_item_table.csv",
        "original_generations": ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
        "perturbed_generations": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa.csv",
        "effects": ROOT / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item.csv",
    },
    "outputs_fixed": {
        "output_dir": ROOT / "outputs",
        "output_name": "factual_paraphrase_item_table_fixed_factual.csv",
        "original_generations": ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
        "perturbed_generations": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "effects": ROOT / "outputs" / "sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv",
    },
    "qwen": {
        "output_dir": ROOT / "qwen" / "outputs",
        "output_name": "factual_paraphrase_item_table.csv",
        "original_generations": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
        "perturbed_generations": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa.csv",
        "effects": ROOT / "qwen" / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item.csv",
    },
    "qwen_fixed": {
        "output_dir": ROOT / "qwen" / "outputs",
        "output_name": "factual_paraphrase_item_table_fixed_factual.csv",
        "original_generations": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_factual_qa.csv",
        "perturbed_generations": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "effects": ROOT / "qwen" / "outputs" / "sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv",
    },
    "llama": {
        "output_dir": ROOT / "llama" / "outputs",
        "output_name": "factual_paraphrase_item_table.csv",
        "original_generations": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_factual_qa.csv",
        "perturbed_generations": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa.csv",
        "effects": ROOT / "llama" / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item.csv",
    },
    "llama_fixed": {
        "output_dir": ROOT / "llama" / "outputs",
        "output_name": "factual_paraphrase_item_table_fixed_factual.csv",
        "original_generations": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_factual_qa.csv",
        "perturbed_generations": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv",
        "effects": ROOT / "llama" / "outputs" / "sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv",
    },
}


def clean_space(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_context_question(prompt: str) -> dict[str, str]:
    """Parse a factual-QA prompt into context and question.

    Uses the first leading `Context:` marker and the last `Question:` marker so
    incidental appearances of the word "question" inside the context are less
    likely to break parsing.
    """

    raw = "" if pd.isna(prompt) else str(prompt).strip()
    if not raw:
        return {"context": "", "question": "", "status": "empty"}

    context_match = re.search(r"(?:^|\n)\s*Context:\s*", raw, flags=re.IGNORECASE)
    question_matches = list(re.finditer(r"(?:^|\n)\s*Question:\s*", raw, flags=re.IGNORECASE))
    if context_match and question_matches:
        question_matches = [match for match in question_matches if match.start() > context_match.end()]
    if context_match and question_matches:
        question_match = question_matches[-1]
        context = raw[context_match.end() : question_match.start()].strip()
        question = raw[question_match.end() :].strip()
        prefix = raw[: context_match.start()].strip()
        if context and question and prefix:
            status = "parsed_full_context_question_with_prefix"
        elif context and question:
            status = "parsed_full_context_question"
        else:
            status = "partial_context_question"
        return {"context": context, "question": question, "status": status}

    question_only = re.match(r"^\s*Question:\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if question_only:
        return {"context": "", "question": question_only.group(1).strip(), "status": "question_only_with_marker"}

    return {"context": "", "question": raw, "status": "question_only_no_marker"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    return pd.read_csv(path)


def unique_text_by_item(df: pd.DataFrame, column: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item_id, group in df.groupby("item_id"):
        unique_values = [value for value in group[column].dropna().astype(str).unique() if value.strip()]
        if unique_values:
            values[item_id] = unique_values[0]
        else:
            values[item_id] = ""
    return values


def sample_counts(df: pd.DataFrame) -> dict[str, int]:
    return df.groupby("item_id")["sample_id"].nunique().astype(int).to_dict()


def load_branch_inputs(branch: str, perturbed_prompts_path: Path) -> dict[str, pd.DataFrame]:
    config = BRANCHES[branch]
    return {
        "original_generations": read_csv(config["original_generations"]),
        "perturbed_generations": read_csv(config["perturbed_generations"]),
        "effects": read_csv(config["effects"]),
        "original_prompts": read_csv(ORIGINAL_PROMPTS),
        "perturbed_prompts": read_csv(perturbed_prompts_path),
    }


def build_branch(branch: str, perturbed_prompts_path: Path) -> pd.DataFrame:
    data = load_branch_inputs(branch, perturbed_prompts_path)
    original_generations = data["original_generations"]
    perturbed_generations = data["perturbed_generations"]
    effects = data["effects"]
    original_prompts = data["original_prompts"]
    perturbed_prompts = data["perturbed_prompts"]

    original_generations = original_generations[original_generations["task_type"] == TASK].copy()
    perturbed_generations = perturbed_generations[
        (perturbed_generations["task_type"] == TASK)
        & (perturbed_generations["perturbation_type"] == PERTURBATION)
    ].copy()
    effects = effects[
        (effects["task_type"] == TASK)
        & (effects["perturbation_type"] == PERTURBATION)
    ].copy()
    original_prompts = original_prompts[original_prompts["task_type"] == TASK].copy()
    perturbed_prompts = perturbed_prompts[
        (perturbed_prompts["task_type"] == TASK)
        & (perturbed_prompts["perturbation_type"] == PERTURBATION)
    ].copy()

    if len(effects) != 50:
        raise SystemExit(f"{branch}: expected 50 factual-QA paraphrasing effect rows, found {len(effects)}")

    original_prompt_from_generation = unique_text_by_item(original_generations, "prompt_text")
    original_prompt_from_prompt_file = unique_text_by_item(original_prompts, "prompt_text")
    perturbed_prompt_from_generation = unique_text_by_item(perturbed_generations, "perturbed_prompt")
    perturbed_prompt_from_prompt_file = unique_text_by_item(perturbed_prompts, "perturbed_prompt")
    original_prompt_in_perturbed_file = unique_text_by_item(perturbed_generations, "original_prompt")

    original_counts = sample_counts(original_generations)
    perturbed_counts = sample_counts(perturbed_generations)
    prompt_metadata = original_prompts.set_index("item_id", drop=False)

    rows: list[dict[str, object]] = []
    for _, effect in effects.sort_values("item_id").iterrows():
        item_id = str(effect["item_id"])
        if item_id not in prompt_metadata.index:
            raise SystemExit(f"{branch}: missing original prompt metadata for {item_id}")

        metadata = prompt_metadata.loc[item_id]
        if isinstance(metadata, pd.DataFrame):
            metadata = metadata.iloc[0]

        original_prompt = original_prompt_from_generation.get(item_id, "") or original_prompt_from_prompt_file.get(item_id, "")
        original_prompt_in_perturbed = original_prompt_in_perturbed_file.get(item_id, "")
        perturbed_prompt = perturbed_prompt_from_generation.get(item_id, "") or perturbed_prompt_from_prompt_file.get(item_id, "")
        perturbed_prompt_in_prompt_file = perturbed_prompt_from_prompt_file.get(item_id, "")

        original_parsed = parse_context_question(original_prompt)
        perturbed_parsed = parse_context_question(perturbed_prompt)

        context = original_parsed["context"]
        original_question = original_parsed["question"]
        paraphrased_question = perturbed_parsed["question"]
        reconstructed = (
            f"Context: {context}\n\nQuestion: {paraphrased_question}"
            if context and paraphrased_question
            else ""
        )

        original_context_norm = clean_space(context)
        perturbed_context_norm = clean_space(perturbed_parsed["context"])
        context_changed = (
            perturbed_parsed["status"] != "parsed_full_context_question"
            or original_context_norm != perturbed_context_norm
        )

        parse_status_parts = [f"original={original_parsed['status']}", f"perturbed={perturbed_parsed['status']}"]
        if not original_question:
            parse_status_parts.append("missing_original_question")
        if not paraphrased_question:
            parse_status_parts.append("missing_paraphrased_question")

        rows.append(
            {
                "branch": branch,
                "item_id": item_id,
                "source_index": metadata.get("source_index", ""),
                "source_id": metadata.get("source_id", ""),
                "reference_answer": metadata.get("reference_answer", ""),
                "original_prompt": original_prompt,
                "perturbed_prompt": perturbed_prompt,
                "perturbed_prompt_raw": perturbed_prompt,
                "reconstructed_perturbed_prompt_with_context": reconstructed,
                "context": context,
                "original_question": original_question,
                "paraphrased_question": paraphrased_question,
                "baseline_similarity": effect["baseline_similarity"],
                "perturbation_similarity": effect["perturbation_similarity"],
                "uncorrected_drift": effect["uncorrected_drift"],
                "noise_corrected_drift": effect["noise_corrected_drift"],
                "n_original_outputs": original_counts.get(item_id, 0),
                "n_perturbed_outputs": perturbed_counts.get(item_id, 0),
                "effects_n_original_outputs": effect["n_original_outputs"],
                "effects_n_perturbed_outputs": effect["n_perturbed_outputs"],
                "original_parse_status": original_parsed["status"],
                "perturbed_parse_status": perturbed_parsed["status"],
                "parse_status": ";".join(parse_status_parts),
                "context_changed": context_changed,
                "original_prompt_matches_generation_in_perturbed_file": clean_space(original_prompt)
                == clean_space(original_prompt_in_perturbed),
                "perturbed_prompt_matches_prompt_file": clean_space(perturbed_prompt)
                == clean_space(perturbed_prompt_in_prompt_file),
            }
        )

    return pd.DataFrame(rows)


def validate_table(branch: str, table: pd.DataFrame) -> None:
    failures: list[str] = []
    if len(table) != 50:
        failures.append(f"expected 50 rows, found {len(table)}")
    if table["item_id"].nunique() != 50:
        failures.append(f"expected 50 unique item_id values, found {table['item_id'].nunique()}")

    checks = {
        "reference_answer": table["reference_answer"].fillna("").astype(str).str.strip().ne("").all(),
        "original_question": table["original_question"].fillna("").astype(str).str.strip().ne("").all(),
        "paraphrased_question": table["paraphrased_question"].fillna("").astype(str).str.strip().ne("").all(),
        "n_original_outputs == 5": table["n_original_outputs"].eq(5).all(),
        "n_perturbed_outputs == 5": table["n_perturbed_outputs"].eq(5).all(),
        "n_original_outputs matches effects": table["n_original_outputs"].eq(table["effects_n_original_outputs"]).all(),
        "n_perturbed_outputs matches effects": table["n_perturbed_outputs"].eq(table["effects_n_perturbed_outputs"]).all(),
    }
    failures.extend(label for label, passed in checks.items() if not passed)

    if failures:
        raise SystemExit(f"{branch}: validation failed: " + "; ".join(failures))


def write_branch(branch: str, perturbed_prompts_path: Path) -> pd.DataFrame:
    table = build_branch(branch, perturbed_prompts_path)
    validate_table(branch, table)
    output_dir = BRANCHES[branch]["output_dir"]
    assert isinstance(output_dir, Path)
    output_name = str(BRANCHES[branch].get("output_name", "factual_paraphrase_item_table.csv"))
    output_path = output_dir / output_name
    table.to_csv(output_path, index=False)
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(
        f"{branch}: rows={len(table)}, context_changed={int(table['context_changed'].sum())}, "
        f"perturbed_parse_status={table['perturbed_parse_status'].value_counts().to_dict()}"
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=["outputs", "outputs_fixed", "qwen", "qwen_fixed", "llama", "llama_fixed", "all"],
        default="outputs",
        help="Branch to build. Use all for outputs, qwen, and llama.",
    )
    parser.add_argument(
        "--perturbed-prompts",
        type=Path,
        default=DEFAULT_PERTURBED_PROMPTS,
        help="Factual-QA perturbed prompt CSV to join for audit columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    perturbed_prompts_path = args.perturbed_prompts
    if not perturbed_prompts_path.is_absolute():
        perturbed_prompts_path = ROOT / perturbed_prompts_path
    for branch in branches:
        write_branch(branch, perturbed_prompts_path)


if __name__ == "__main__":
    main()
