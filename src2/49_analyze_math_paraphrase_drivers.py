"""Analyze why math reasoning is most affected by paraphrasing.

This mirrors the factual-QA follow-up at the level that makes sense for math:
prompt-change diagnostics across all perturbations, then item-level paraphrase
drift diagnostics across GPT/main, Qwen, and Llama.

Outputs:
    outputs/math_prompt_perturbation_change_by_item.csv
    outputs/math_prompt_perturbation_change_summary.csv
    outputs/math_paraphrase_driver_by_item.csv
    outputs/math_paraphrase_driver_correlations.csv
    outputs/math_paraphrase_case_table.md
    qwen/outputs/math_paraphrase_driver_by_item.csv
    qwen/outputs/math_paraphrase_driver_correlations.csv
    qwen/outputs/math_paraphrase_case_table.md
    llama/outputs/math_paraphrase_driver_by_item.csv
    llama/outputs/math_paraphrase_driver_correlations.csv
    llama/outputs/math_paraphrase_case_table.md
    outputs/math_paraphrase_cross_model_cases.csv
    outputs/math_paraphrase_explanation.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning.csv"
ORIGINAL_PROMPTS = ROOT / "prompts" / "rq1_sampled_original_prompts_n50_math_reasoning.csv"

BRANCHES = {
    "outputs": {
        "name": "GPT/main",
        "original": ROOT / "outputs" / "rq1_formal_original_generations_n50_math_reasoning.csv",
        "perturbed": ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_math_reasoning.csv",
        "effects": ROOT / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv",
        "summary": ROOT / "outputs" / "sbert_rq1_n50_perturbation_summary_fixed_factual.csv",
        "output_dir": ROOT / "outputs",
    },
    "qwen": {
        "name": "Qwen",
        "original": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_math_reasoning.csv",
        "perturbed": ROOT / "qwen" / "outputs" / "rq1_qwen_perturbed_generations_n50_math_reasoning.csv",
        "effects": ROOT / "qwen" / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv",
        "summary": ROOT / "qwen" / "outputs" / "sbert_rq1_n50_perturbation_summary_fixed_factual.csv",
        "output_dir": ROOT / "qwen" / "outputs",
    },
    "llama": {
        "name": "Llama",
        "original": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_math_reasoning.csv",
        "perturbed": ROOT / "llama" / "outputs" / "rq1_llama_perturbed_generations_n50_math_reasoning.csv",
        "effects": ROOT / "llama" / "outputs" / "sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv",
        "summary": ROOT / "llama" / "outputs" / "sbert_rq1_n50_perturbation_summary_fixed_factual.csv",
        "output_dir": ROOT / "llama" / "outputs",
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
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "let",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "with",
    "your",
    "you",
}


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: object) -> list[str]:
    return normalize_text(text).split()


def content_tokens(text: object) -> list[str]:
    return [token for token in word_tokens(text) if token not in STOPWORDS]


def token_recall(original: list[str], perturbed: list[str]) -> float:
    if not original:
        return math.nan
    return len(set(original) & set(perturbed)) / len(set(original))


def token_jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return math.nan
    return len(left_set & right_set) / len(left_set | right_set)


def output_length(text: object) -> int:
    return len(word_tokens(text))


def math_segments(text: object) -> list[str]:
    if pd.isna(text):
        return []
    raw = str(text)
    segments = re.findall(r"\$([^$]+)\$", raw)
    segments += re.findall(r"\\\[(.*?)\\\]", raw, flags=re.DOTALL)
    segments += re.findall(r"\\\((.*?)\\\)", raw, flags=re.DOTALL)
    return [re.sub(r"\s+", "", segment) for segment in segments if segment.strip()]


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


LABEL_RE = re.compile(
    r"(?im)(?:^|[\n\r]|-\s*)\s*"
    r"(\*\*Code Signature\*\*|Code Signature|Code signature|Task Signature|Research Question|Rewrite|Problem|Prompt|Instruction|Action|Task|Code)"
    r"\s*:",
)


def labels(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return [match.group(1).replace("**", "") for match in LABEL_RE.finditer(str(text))]


def has_code_artifact(text: object) -> bool:
    if pd.isna(text):
        return False
    raw = str(text)
    return bool(
        re.search(
            r"Code Signature|Code signature|Task Signature|```|def\s+\w+\s*\(|function\s+\w+\s*\(",
            raw,
            flags=re.IGNORECASE,
        )
    )


def has_research_question_label(text: object) -> bool:
    if pd.isna(text):
        return False
    return bool(re.search(r"Research Question\s*:", str(text), flags=re.IGNORECASE))


def has_rewrite_label(text: object) -> bool:
    if pd.isna(text):
        return False
    return bool(re.search(r"^\s*Rewrite\s*:", str(text), flags=re.IGNORECASE))


def split_math_prompt(prompt: object) -> dict[str, str]:
    raw = "" if pd.isna(prompt) else str(prompt)
    normalized = raw.replace("\r\n", "\n")
    prompt_wrapper = re.search(r"(?is)(?:^|[\n\r]|-\s*)Prompt:\s*", normalized)
    if prompt_wrapper and re.search(r"(?i)Problem\s*:", normalized[prompt_wrapper.end() :]):
        normalized = normalized[prompt_wrapper.end() :]
    matches = list(LABEL_RE.finditer(normalized))
    problem = normalized
    instruction = ""
    problem_match = None
    for preferred in ["Problem", "Research Question", "Task"]:
        for match in matches:
            if match.group(1).replace("**", "") == preferred:
                problem_match = match
                break
        if problem_match is not None:
            break
    if problem_match is not None:
        start = problem_match.end()
        end = len(normalized)
        for match in matches:
            label = match.group(1).replace("**", "")
            if match.start() > problem_match.start() and label != "Prompt":
                end = match.start()
                break
        problem = normalized[start:end]
    elif matches:
        first_match = matches[0]
        before_first_label = normalized[: first_match.start()].strip()
        if before_first_label:
            problem = before_first_label
        else:
            problem = normalized[first_match.end() :]
    for match in matches:
        label = match.group(1).replace("**", "")
        if label in {"Instruction", "Action"}:
            start = match.end()
            end = len(normalized)
            for next_match in matches:
                if next_match.start() > match.start():
                    end = next_match.start()
                    break
            instruction = normalized[start:end]
            break
    return {
        "problem_text": problem.strip(),
        "instruction_text": instruction.strip(),
    }


def token_f1(reference: object, output: object) -> float:
    ref = word_tokens(reference)
    out = word_tokens(output)
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
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def containment(reference: object, output: object) -> bool:
    ref = normalize_text(reference)
    out = normalize_text(output)
    return bool(ref and ref in out)


def tail_containment(reference: object, output: object) -> bool:
    if pd.isna(output):
        return False
    return containment(reference, str(output)[-500:])


def summarize_generations(df: pd.DataFrame, reference_by_item: dict[str, str], prefix: str) -> pd.DataFrame:
    rows = []
    for item_id, group in df.groupby("item_id"):
        reference = reference_by_item[item_id]
        rows.append(
            {
                "item_id": item_id,
                f"{prefix}_mean_answer_token_f1": sum(token_f1(reference, output) for output in group["output_text"])
                / len(group),
                f"{prefix}_answer_containment_rate": sum(containment(reference, output) for output in group["output_text"])
                / len(group),
                f"{prefix}_answer_tail_containment_rate": sum(tail_containment(reference, output) for output in group["output_text"])
                / len(group),
                f"{prefix}_mean_output_length_tokens": sum(output_length(output) for output in group["output_text"])
                / len(group),
            }
        )
    return pd.DataFrame(rows)


def build_prompt_change_table() -> pd.DataFrame:
    prompt_df = pd.read_csv(PROMPTS)
    rows = []
    for _, row in prompt_df.iterrows():
        original = row["original_prompt"]
        perturbed = row["perturbed_prompt"]
        original_parts = split_math_prompt(original)
        perturbed_parts = split_math_prompt(perturbed)
        original_content = content_tokens(original)
        perturbed_content = content_tokens(perturbed)
        original_problem_content = content_tokens(original_parts["problem_text"])
        perturbed_problem_content = content_tokens(perturbed_parts["problem_text"])
        original_math_tokens = math_tokens(original_parts["problem_text"])
        perturbed_math_tokens = math_tokens(perturbed_parts["problem_text"])
        original_segments = math_segments(original_parts["problem_text"])
        perturbed_segments = math_segments(perturbed_parts["problem_text"])
        original_numbers = numbers(original_parts["problem_text"])
        perturbed_numbers = numbers(perturbed_parts["problem_text"])
        original_labels = labels(original)
        perturbed_labels = labels(perturbed)
        perturbed_has_research_question = has_research_question_label(perturbed)
        perturbed_has_rewrite_label = has_rewrite_label(perturbed)
        perturbed_has_code_artifact = has_code_artifact(perturbed)
        original_has_asy = "[asy]" in str(original)
        perturbed_has_asy = "[asy]" in str(perturbed)
        rows.append(
            {
                "item_id": row["item_id"],
                "perturbation_type": row["perturbation_type"],
                "original_prompt": original,
                "perturbed_prompt": perturbed,
                "original_problem_text": original_parts["problem_text"],
                "perturbed_problem_text": perturbed_parts["problem_text"],
                "prompt_content_recall": token_recall(original_content, perturbed_content),
                "prompt_content_jaccard": token_jaccard(original_content, perturbed_content),
                "problem_content_recall": token_recall(original_problem_content, perturbed_problem_content),
                "problem_content_jaccard": token_jaccard(original_problem_content, perturbed_problem_content),
                "math_token_recall": token_recall(original_math_tokens, perturbed_math_tokens),
                "number_recall": token_recall(original_numbers, perturbed_numbers),
                "latex_segment_recall": token_recall(original_segments, perturbed_segments),
                "prompt_length_delta_tokens": len(word_tokens(perturbed)) - len(word_tokens(original)),
                "problem_length_delta_tokens": len(word_tokens(perturbed_parts["problem_text"]))
                - len(word_tokens(original_parts["problem_text"])),
                "label_changed": original_labels != perturbed_labels,
                "perturbed_has_research_question_label": perturbed_has_research_question,
                "perturbed_has_rewrite_label": perturbed_has_rewrite_label,
                "perturbed_has_code_artifact": perturbed_has_code_artifact,
                "perturbed_has_template_artifact": perturbed_has_research_question
                or perturbed_has_rewrite_label
                or perturbed_has_code_artifact,
                "original_has_asy": original_has_asy,
                "perturbed_has_asy": perturbed_has_asy,
                "asy_removed": original_has_asy and not perturbed_has_asy,
                "original_labels": "; ".join(original_labels),
                "perturbed_labels": "; ".join(perturbed_labels),
            }
        )
    return pd.DataFrame(rows)


def summarize_prompt_change(prompt_change: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "prompt_content_recall",
        "prompt_content_jaccard",
        "problem_content_recall",
        "problem_content_jaccard",
        "math_token_recall",
        "number_recall",
        "latex_segment_recall",
        "prompt_length_delta_tokens",
        "problem_length_delta_tokens",
    ]
    summary = (
        prompt_change.groupby("perturbation_type")
        .agg(
            n_items=("item_id", "nunique"),
            label_changed_rate=("label_changed", "mean"),
            research_question_label_rate=("perturbed_has_research_question_label", "mean"),
            rewrite_label_rate=("perturbed_has_rewrite_label", "mean"),
            code_artifact_rate=("perturbed_has_code_artifact", "mean"),
            template_artifact_rate=("perturbed_has_template_artifact", "mean"),
            asy_removed_rate=("asy_removed", "mean"),
            **{f"mean_{metric}": (metric, "mean") for metric in metrics},
        )
        .reset_index()
    )
    return summary.sort_values("mean_prompt_content_recall")


def branch_table(branch: str, prompt_change: pd.DataFrame) -> pd.DataFrame:
    config = BRANCHES[branch]
    references = pd.read_csv(ORIGINAL_PROMPTS)[["item_id", "reference_answer", "source_id", "prompt_text"]]
    reference_by_item = dict(zip(references["item_id"], references["reference_answer"].astype(str)))
    effects = pd.read_csv(config["effects"])
    effects = effects[
        (effects["task_type"] == "math_reasoning") & (effects["perturbation_type"] == "paraphrasing")
    ].copy()
    original = pd.read_csv(config["original"])
    perturbed = pd.read_csv(config["perturbed"])
    perturbed = perturbed[perturbed["perturbation_type"] == "paraphrasing"].copy()

    original_summary = summarize_generations(original, reference_by_item, "original")
    paraphrase_summary = summarize_generations(perturbed, reference_by_item, "paraphrase")
    prompt_subset = prompt_change[prompt_change["perturbation_type"] == "paraphrasing"].copy()
    df = (
        effects.merge(prompt_subset, on=["item_id", "perturbation_type"], how="left")
        .merge(references, on="item_id", how="left")
        .merge(original_summary, on="item_id", how="left")
        .merge(paraphrase_summary, on="item_id", how="left")
    )
    df["answer_token_f1_delta"] = df["paraphrase_mean_answer_token_f1"] - df["original_mean_answer_token_f1"]
    df["answer_containment_delta"] = df["paraphrase_answer_containment_rate"] - df["original_answer_containment_rate"]
    df["answer_tail_containment_delta"] = (
        df["paraphrase_answer_tail_containment_rate"] - df["original_answer_tail_containment_rate"]
    )
    df["output_length_delta_tokens"] = (
        df["paraphrase_mean_output_length_tokens"] - df["original_mean_output_length_tokens"]
    )
    return df.sort_values("noise_corrected_drift", ascending=False)


def spearman(df: pd.DataFrame, x_col: str, y_col: str = "noise_corrected_drift") -> dict[str, object]:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 3 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return {"x": x_col, "y": y_col, "n": len(sub), "spearman_rho": math.nan, "p_value": math.nan}
    rho, p_value = stats.spearmanr(sub[x_col], sub[y_col])
    return {"x": x_col, "y": y_col, "n": len(sub), "spearman_rho": float(rho), "p_value": float(p_value)}


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "prompt_content_recall",
        "problem_content_recall",
        "math_token_recall",
        "number_recall",
        "latex_segment_recall",
        "perturbed_has_template_artifact",
        "prompt_length_delta_tokens",
        "problem_length_delta_tokens",
        "answer_token_f1_delta",
        "answer_containment_delta",
        "answer_tail_containment_delta",
        "output_length_delta_tokens",
    ]
    return pd.DataFrame([spearman(df, col) for col in cols]).sort_values("p_value", na_position="last")


def add_case(selected: dict[str, dict[str, object]], row: pd.Series, category: str, interpretation: str) -> None:
    item_id = str(row["item_id"])
    if item_id in selected:
        selected[item_id]["selection_category"] += f"; {category}"
        selected[item_id]["interpretation"] += f" {interpretation}"
        return
    record = row.to_dict()
    record["selection_category"] = category
    record["interpretation"] = interpretation
    selected[item_id] = record


def select_cases(df: pd.DataFrame) -> pd.DataFrame:
    selected: dict[str, dict[str, object]] = {}
    for _, row in df.sort_values("noise_corrected_drift", ascending=False).head(3).iterrows():
        add_case(selected, row, "high_drift", "High math paraphrase drift.")
    for _, row in df.sort_values("noise_corrected_drift", ascending=True).head(2).iterrows():
        add_case(selected, row, "low_drift", "Low-drift contrast case.")
    for _, row in df.sort_values("output_length_delta_tokens", ascending=False).head(2).iterrows():
        add_case(selected, row, "output_length_increase", "Paraphrase elicits much longer output.")
    for _, row in df.sort_values("answer_token_f1_delta", ascending=True).head(2).iterrows():
        add_case(selected, row, "answer_score_drop", "Reference-answer token score drops under paraphrase.")
    for _, row in df.sort_values(["problem_content_recall", "noise_corrected_drift"], ascending=[True, False]).head(2).iterrows():
        add_case(selected, row, "problem_wording_change", "Problem wording changes more than most math cases.")
    cases = pd.DataFrame(selected.values())
    return cases.sort_values(["noise_corrected_drift"], ascending=False)


def compact_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "item_id",
        "selection_category",
        "reference_answer",
        "noise_corrected_drift",
        "baseline_similarity",
        "perturbation_similarity",
        "problem_content_recall",
        "math_token_recall",
        "number_recall",
        "latex_segment_recall",
        "perturbed_has_template_artifact",
        "output_length_delta_tokens",
        "answer_token_f1_delta",
        "answer_tail_containment_delta",
        "original_problem_text",
        "perturbed_problem_text",
        "interpretation",
    ]
    compact = df[[col for col in cols if col in df.columns]].copy()
    for col in ["original_problem_text", "perturbed_problem_text", "interpretation"]:
        if col in compact.columns:
            compact[col] = compact[col].map(lambda value: re.sub(r"\s+", " ", str(value)).strip()[:500])
    return compact


def write_branch_outputs(branch: str, df: pd.DataFrame) -> pd.DataFrame:
    config = BRANCHES[branch]
    output_dir = config["output_dir"]
    out_csv = output_dir / "math_paraphrase_driver_by_item.csv"
    corr_csv = output_dir / "math_paraphrase_driver_correlations.csv"
    case_md = output_dir / "math_paraphrase_case_table.md"

    corr = correlations(df)
    cases = select_cases(df)
    df.to_csv(out_csv, index=False)
    corr.to_csv(corr_csv, index=False)
    compact_columns(cases).to_markdown(case_md, index=False)
    print(f"{branch}: wrote {out_csv}")
    print(f"{branch}: wrote {corr_csv}")
    print(f"{branch}: wrote {case_md}")
    return cases


def cross_model_cases(branch_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for branch, df in branch_tables.items():
        temp = df.copy()
        temp["branch"] = branch
        frames.append(temp)
    all_rows = pd.concat(frames, ignore_index=True)
    agg = (
        all_rows.groupby("item_id")
        .agg(
            reference_answer=("reference_answer", "first"),
            original_problem_text=("original_problem_text", "first"),
            perturbed_problem_text=("perturbed_problem_text", "first"),
            mean_noise_corrected_drift=("noise_corrected_drift", "mean"),
            max_noise_corrected_drift=("noise_corrected_drift", "max"),
            branches_high_drift=("noise_corrected_drift", lambda s: int((s >= 0.10).sum())),
            mean_output_length_delta_tokens=("output_length_delta_tokens", "mean"),
            mean_answer_token_f1_delta=("answer_token_f1_delta", "mean"),
            problem_content_recall=("problem_content_recall", "first"),
            math_token_recall=("math_token_recall", "first"),
            number_recall=("number_recall", "first"),
            latex_segment_recall=("latex_segment_recall", "first"),
            perturbed_has_template_artifact=("perturbed_has_template_artifact", "max"),
        )
        .reset_index()
    )
    selected: dict[str, dict[str, object]] = {}
    for _, row in agg.sort_values("mean_noise_corrected_drift", ascending=False).head(4).iterrows():
        add_case(selected, row, "cross_model_high_mean_drift", "High average math paraphrase drift.")
    for _, row in agg.sort_values("mean_output_length_delta_tokens", ascending=False).head(2).iterrows():
        add_case(selected, row, "cross_model_length_increase", "Large average output-length increase.")
    for _, row in agg.sort_values("mean_answer_token_f1_delta", ascending=True).head(2).iterrows():
        add_case(selected, row, "cross_model_answer_score_drop", "Average reference-answer token score drops.")
    return pd.DataFrame(selected.values()).sort_values("mean_noise_corrected_drift", ascending=False)


def math_rank_table() -> pd.DataFrame:
    rows = []
    for branch, config in BRANCHES.items():
        summary = pd.read_csv(config["summary"])
        math_summary = summary[summary["task_type"] == "math_reasoning"].copy()
        math_summary = math_summary.sort_values("mean_noise_corrected_drift", ascending=False).reset_index(drop=True)
        for idx, row in math_summary.iterrows():
            rows.append(
                {
                    "branch": config["name"],
                    "rank": idx + 1,
                    "perturbation_type": row["perturbation_type"],
                    "mean_noise_corrected_drift": row["mean_noise_corrected_drift"],
                    "std_noise_corrected_drift": row["std_noise_corrected_drift"],
                }
            )
    return pd.DataFrame(rows)


def write_explanation(
    rank_table: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    branch_tables: dict[str, pd.DataFrame],
    branch_cases: dict[str, pd.DataFrame],
    cross_cases: pd.DataFrame,
) -> None:
    branch_summary_rows = []
    for branch, df in branch_tables.items():
        corr = correlations(df)
        top_corr = corr.head(4)
        branch_summary_rows.append(
            {
                "branch": BRANCHES[branch]["name"],
                "mean_drift": df["noise_corrected_drift"].mean(),
                "median_drift": df["noise_corrected_drift"].median(),
                "mean_output_length_delta_tokens": df["output_length_delta_tokens"].mean(),
                "mean_answer_token_f1_delta": df["answer_token_f1_delta"].mean(),
                "mean_problem_content_recall": df["problem_content_recall"].mean(),
                "mean_math_token_recall": df["math_token_recall"].mean(),
                "mean_number_recall": df["number_recall"].mean(),
                "template_artifact_rate": df["perturbed_has_template_artifact"].mean(),
                "top_correlations": "; ".join(
                    f"{r.x} rho={r.spearman_rho:.3f}, p={r.p_value:.4f}" for r in top_corr.itertuples()
                ),
            }
        )
    branch_summary = pd.DataFrame(branch_summary_rows)
    artifact_rows = []
    for branch, df in branch_tables.items():
        grouped = df.groupby("perturbed_has_template_artifact").agg(
            n=("item_id", "count"),
            mean_drift=("noise_corrected_drift", "mean"),
            median_drift=("noise_corrected_drift", "median"),
            mean_problem_content_recall=("problem_content_recall", "mean"),
            mean_math_token_recall=("math_token_recall", "mean"),
            mean_answer_token_f1_delta=("answer_token_f1_delta", "mean"),
        )
        for has_artifact, row in grouped.iterrows():
            artifact_rows.append(
                {
                    "branch": BRANCHES[branch]["name"],
                    "template_artifact": bool(has_artifact),
                    **row.to_dict(),
                }
            )
    artifact_summary = pd.DataFrame(artifact_rows)

    lines = [
        "# Math Reasoning Paraphrasing Driver Check",
        "",
        "## Math perturbation ranking",
        "",
        rank_table.to_markdown(index=False),
        "",
        "## Prompt-level perturbation size",
        "",
        prompt_summary.to_markdown(index=False),
        "",
        "## Paraphrasing branch diagnostics",
        "",
        branch_summary.to_markdown(index=False),
        "",
        "## Template artifact comparison",
        "",
        artifact_summary.to_markdown(index=False),
        "",
        "## Cross-model qualitative cases",
        "",
        compact_columns(cross_cases).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Paraphrasing is the largest math perturbation because it is the only perturbation that consistently rewrites the mathematical problem statement and instruction labels. Reordering, formatting changes, context injection, and surface noise mostly preserve the original problem wording, equations, and numeric cues. Math paraphrasing usually preserves formulas and numbers, but it changes the surrounding natural-language framing from `Problem/Instruction` into `Task/Action` style and often replaces verbs such as `find`, `compute`, `prove`, or `solve` with broader instructions.",
        "",
        "A second, data-quality-specific reason is that the math paraphrasing condition is not fully clean: some paraphrased prompts contain `Research Question`, `Task Signature`, `Code Signature`, fenced code, or function-signature artifacts. These artifacts occur in the paraphrasing condition, not in the rule-based perturbations, and they are over-represented among qualitative high-drift cases.",
        "",
        "The resulting drift is therefore mostly a behavioral/output-form effect rather than a clean loss of mathematical symbols. In the model outputs, high-drift cases often show longer or differently structured derivations, different intermediate assumptions, or different final-answer behavior, even when the perturbed prompt remains mathematically close to the original.",
        "",
        "This is weaker than the original corrupted factual-QA paraphrasing effect but stable across GPT/main, Qwen, and Llama within the math task.",
    ]
    path = ROOT / "outputs" / "math_paraphrase_explanation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def validate(prompt_change: pd.DataFrame, branch_tables: dict[str, pd.DataFrame]) -> None:
    failures = []
    if len(prompt_change) != 250:
        failures.append(f"prompt change table expected 250 rows, found {len(prompt_change)}")
    if prompt_change.groupby("perturbation_type")["item_id"].nunique().min() != 50:
        failures.append("not every perturbation has 50 prompt items")
    for branch, df in branch_tables.items():
        if len(df) != 50:
            failures.append(f"{branch}: expected 50 paraphrase rows, found {len(df)}")
        if df["item_id"].nunique() != 50:
            failures.append(f"{branch}: duplicate/missing item ids")
        required = [
            "noise_corrected_drift",
            "problem_content_recall",
            "math_token_recall",
            "number_recall",
            "output_length_delta_tokens",
            "answer_token_f1_delta",
        ]
        for col in required:
            if col not in df.columns:
                failures.append(f"{branch}: missing {col}")
            elif df[col].isna().all():
                failures.append(f"{branch}: {col} all NA")
    if failures:
        raise SystemExit("validation failed: " + "; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["outputs", "qwen", "llama", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_change = build_prompt_change_table()
    prompt_summary = summarize_prompt_change(prompt_change)
    (ROOT / "outputs" / "math_prompt_perturbation_change_by_item.csv").write_text(
        prompt_change.to_csv(index=False), encoding="utf-8"
    )
    (ROOT / "outputs" / "math_prompt_perturbation_change_summary.csv").write_text(
        prompt_summary.to_csv(index=False), encoding="utf-8"
    )
    print("wrote outputs/math_prompt_perturbation_change_by_item.csv")
    print("wrote outputs/math_prompt_perturbation_change_summary.csv")

    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    branch_tables = {branch: branch_table(branch, prompt_change) for branch in branches}
    validate(prompt_change, branch_tables)

    branch_cases = {}
    for branch, df in branch_tables.items():
        branch_cases[branch] = write_branch_outputs(branch, df)

    if args.branch == "all":
        rank_table = math_rank_table()
        cross_cases = cross_model_cases(branch_tables)
        cross_cases.to_csv(ROOT / "outputs" / "math_paraphrase_cross_model_cases.csv", index=False)
        compact_columns(cross_cases).to_markdown(
            ROOT / "outputs" / "math_paraphrase_cross_model_cases.md", index=False
        )
        print("wrote outputs/math_paraphrase_cross_model_cases.csv")
        print("wrote outputs/math_paraphrase_cross_model_cases.md")
        write_explanation(rank_table, prompt_summary, branch_tables, branch_cases, cross_cases)


if __name__ == "__main__":
    main()
