"""Explain why fixed math paraphrasing remains the largest math drift cell."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv"
ORIGINAL_PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50_math_reasoning.csv"

BRANCHES = {
    "GPT/main": {
        "base": ROOT / "outputs",
        "original": ROOT / "outputs" / "rq1_formal_original_generations_n50_math_reasoning.csv",
        "fixed_paraphrase": ROOT
        / "outputs"
        / "rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
    },
    "Qwen": {
        "base": ROOT / "qwen" / "outputs",
        "original": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_math_reasoning.csv",
        "fixed_paraphrase": ROOT
        / "qwen"
        / "outputs"
        / "rq1_qwen_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
    },
    "Llama": {
        "base": ROOT / "llama" / "outputs",
        "original": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_math_reasoning.csv",
        "fixed_paraphrase": ROOT
        / "llama"
        / "outputs"
        / "rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
    },
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "given",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "with",
}


def normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: object) -> list[str]:
    return normalize(text).split()


def content_words(text: object) -> list[str]:
    return [token for token in words(text) if token not in STOPWORDS]


def recall(original: list[str], perturbed: list[str]) -> float:
    original_set = set(original)
    perturbed_set = set(perturbed)
    if not original_set:
        return math.nan
    return len(original_set & perturbed_set) / len(original_set)


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return math.nan
    return len(left_set & right_set) / len(left_set | right_set)


def extract_problem(prompt: object) -> str:
    text = "" if pd.isna(prompt) else str(prompt)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_wrapper = re.search(r"(?is)(?:^|[\n\r]|-\s*)Prompt:\s*", text)
    if prompt_wrapper and re.search(r"(?i)Problem\s*:", text[prompt_wrapper.end() :]):
        text = text[prompt_wrapper.end() :]
    problem_match = re.search(r"(?im)(?:^|[\n\r])\s*Problem\s*:\s*", "\n" + text)
    if problem_match:
        text = ("\n" + text)[problem_match.end() :]
    text = re.split(r"(?im)^\s*Instruction\s*:\s*", text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text).strip()


def math_tokens(text: object) -> list[str]:
    if pd.isna(text):
        return []
    raw = str(text)
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|\\[A-Za-z]+|[=+\-*/^_{}|<>()[\],]", raw)
    return [token for token in tokens if re.search(r"\d|\\|[=+\-*/^_{}|<>]", token)]


def numbers(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return re.findall(r"-?\d+(?:\.\d+)?", str(text))


def output_len(text: object) -> int:
    return len(words(text))


def token_f1(reference: object, output: object) -> float:
    ref = words(reference)
    out = words(output)
    if not ref or not out:
        return 0.0
    counts: dict[str, int] = {}
    for token in ref:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in out:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(out)
    recall_value = overlap / len(ref)
    return 2 * precision * recall_value / (precision + recall_value)


def contains(reference: object, output: object) -> bool:
    ref = normalize(reference)
    out = normalize(output)
    return bool(ref and ref in out)


def prompt_change_table() -> pd.DataFrame:
    df = pd.read_csv(PROMPTS)
    rows = []
    for _, row in df.iterrows():
        original_problem = extract_problem(row["original_prompt"])
        perturbed_problem = extract_problem(row["perturbed_prompt"])
        rows.append(
            {
                "item_id": row["item_id"],
                "perturbation_type": row["perturbation_type"],
                "problem_content_recall": recall(content_words(original_problem), content_words(perturbed_problem)),
                "problem_content_jaccard": jaccard(content_words(original_problem), content_words(perturbed_problem)),
                "prompt_content_recall": recall(content_words(row["original_prompt"]), content_words(row["perturbed_prompt"])),
                "math_token_recall": recall(math_tokens(original_problem), math_tokens(perturbed_problem)),
                "number_recall": recall(numbers(original_problem), numbers(perturbed_problem)),
                "problem_length_delta_tokens": len(words(perturbed_problem)) - len(words(original_problem)),
            }
        )
    return pd.DataFrame(rows)


def generation_summary(path: Path, reference_by_item: dict[str, str], prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for item_id, group in df.groupby("item_id"):
        reference = reference_by_item[item_id]
        rows.append(
            {
                "item_id": item_id,
                f"{prefix}_mean_output_length_tokens": group["output_text"].map(output_len).mean(),
                f"{prefix}_answer_token_f1": group["output_text"].map(lambda out: token_f1(reference, out)).mean(),
                f"{prefix}_answer_containment": group["output_text"].map(lambda out: contains(reference, out)).mean(),
            }
        )
    return pd.DataFrame(rows)


def spearman(df: pd.DataFrame, x: str, y: str = "noise_corrected_drift") -> dict[str, object]:
    sub = df[[x, y]].dropna()
    if len(sub) < 3 or sub[x].nunique() < 2:
        return {"x": x, "y": y, "n": len(sub), "rho": math.nan, "p": math.nan}
    rho, p = stats.spearmanr(sub[x], sub[y])
    return {"x": x, "y": y, "n": len(sub), "rho": float(rho), "p": float(p)}


def main() -> None:
    prompt_rows = prompt_change_table()
    prompt_summary = (
        prompt_rows.groupby("perturbation_type")
        .agg(
            n_items=("item_id", "nunique"),
            mean_problem_content_recall=("problem_content_recall", "mean"),
            mean_problem_content_jaccard=("problem_content_jaccard", "mean"),
            mean_prompt_content_recall=("prompt_content_recall", "mean"),
            mean_math_token_recall=("math_token_recall", "mean"),
            mean_number_recall=("number_recall", "mean"),
            mean_problem_length_delta_tokens=("problem_length_delta_tokens", "mean"),
        )
        .reset_index()
        .sort_values("mean_problem_content_recall")
    )
    prompt_rows.to_csv(ROOT / "outputs" / "math_fixed_prompt_change_by_item.csv", index=False)
    prompt_summary.to_csv(ROOT / "outputs" / "math_fixed_prompt_change_summary.csv", index=False)

    refs = pd.read_csv(ORIGINAL_PROMPTS)
    reference_by_item = dict(zip(refs["item_id"], refs["reference_answer"].astype(str)))
    branch_summaries = []
    corr_rows = []
    for branch, config in BRANCHES.items():
        effects = pd.read_csv(config["base"] / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual_math.csv")
        effects = effects[
            (effects["task_type"] == "math_reasoning")
            & (effects["perturbation_type"] == "paraphrasing")
        ].copy()
        original_summary = generation_summary(config["original"], reference_by_item, "original")
        para_summary = generation_summary(config["fixed_paraphrase"], reference_by_item, "paraphrase")
        driver = (
            effects.merge(prompt_rows[prompt_rows["perturbation_type"] == "paraphrasing"], on=["item_id", "perturbation_type"])
            .merge(original_summary, on="item_id")
            .merge(para_summary, on="item_id")
        )
        driver["output_length_delta_tokens"] = (
            driver["paraphrase_mean_output_length_tokens"] - driver["original_mean_output_length_tokens"]
        )
        driver["answer_token_f1_delta"] = driver["paraphrase_answer_token_f1"] - driver["original_answer_token_f1"]
        driver["answer_containment_delta"] = (
            driver["paraphrase_answer_containment"] - driver["original_answer_containment"]
        )
        driver.to_csv(config["base"] / "math_fixed_paraphrase_driver_by_item.csv", index=False)
        for metric in [
            "problem_content_recall",
            "prompt_content_recall",
            "math_token_recall",
            "number_recall",
            "problem_length_delta_tokens",
            "output_length_delta_tokens",
            "answer_token_f1_delta",
            "answer_containment_delta",
        ]:
            corr_rows.append({"branch": branch, **spearman(driver, metric)})
        branch_summaries.append(
            {
                "branch": branch,
                "mean_ncp": driver["noise_corrected_drift"].mean(),
                "median_ncp": driver["noise_corrected_drift"].median(),
                "mean_problem_content_recall": driver["problem_content_recall"].mean(),
                "mean_math_token_recall": driver["math_token_recall"].mean(),
                "mean_number_recall": driver["number_recall"].mean(),
                "mean_output_length_delta_tokens": driver["output_length_delta_tokens"].mean(),
                "mean_answer_token_f1_delta": driver["answer_token_f1_delta"].mean(),
                "mean_answer_containment_delta": driver["answer_containment_delta"].mean(),
            }
        )

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(ROOT / "outputs" / "math_fixed_paraphrase_driver_correlations.csv", index=False)
    branch_summary = pd.DataFrame(branch_summaries)
    branch_summary.to_csv(ROOT / "outputs" / "math_fixed_paraphrase_driver_summary.csv", index=False)

    lines = [
        "# Fixed Math Paraphrasing Drift Reason",
        "",
        "## Fixed prompt-change comparison",
        "",
        prompt_summary.to_markdown(index=False),
        "",
        "## Fixed paraphrase branch diagnostics",
        "",
        branch_summary.to_markdown(index=False),
        "",
        "## Correlations with fixed paraphrase drift",
        "",
        corr.sort_values(["branch", "p"]).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "After repair, paraphrasing remains the largest math perturbation because it is still the only perturbation that changes the mathematical problem wording at scale. Reordering, formatting changes, and context injection preserve the problem text, math tokens, and numbers almost exactly; surface noise makes small local edits. Fixed paraphrasing removes template artifacts and restores graphs, but it still changes problem phrasing, instruction framing, and some symbolic/numeric surface cues.",
        "",
        "The strongest quantitative reason is prompt-level cue change: fixed paraphrasing has the lowest problem-content recall and math-token/number recall among the clean perturbations. Across branches, larger drift is generally associated with lower math-token or problem-content preservation and with changes in answer/output behavior.",
    ]
    report = ROOT / "outputs" / "math_fixed_paraphrase_reason.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
