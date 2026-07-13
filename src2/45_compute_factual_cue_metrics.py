"""Compute factual-QA paraphrase cue preservation metrics.

This script uses the repaired context-preserving factual QA paraphrase tables.
It computes both cue preservation metrics and question-context overlap metrics.

Outputs:
    outputs/factual_paraphrase_cue_metrics_fixed_factual.csv
    qwen/outputs/factual_paraphrase_cue_metrics_fixed_factual.csv
    llama/outputs/factual_paraphrase_cue_metrics_fixed_factual.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BRANCHES = {
    "outputs": {
        "input": ROOT / "outputs" / "factual_paraphrase_item_table_fixed_factual.csv",
        "output": ROOT / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
    },
    "qwen": {
        "input": ROOT / "qwen" / "outputs" / "factual_paraphrase_item_table_fixed_factual.csv",
        "output": ROOT / "qwen" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
    },
    "llama": {
        "input": ROOT / "llama" / "outputs" / "factual_paraphrase_item_table_fixed_factual.csv",
        "output": ROOT / "llama" / "outputs" / "factual_paraphrase_cue_metrics_fixed_factual.csv",
    },
}

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

WH_WORDS = ["who", "what", "when", "where", "why", "how", "which", "whose"]


def text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def content_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    kept = []
    for token in raw_tokens:
        lower = token.lower()
        if lower not in STOPWORDS or token[:1].isupper() or any(ch.isdigit() for ch in token):
            kept.append(lower)
    return kept


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return math.nan
    union = left | right
    return len(left & right) / len(union) if union else math.nan


def recall(original: set[str], paraphrase: set[str]) -> float:
    if not original:
        return math.nan
    return len(original & paraphrase) / len(original)


def first_wh_word(text: str) -> str:
    lowered = tokens(text)
    for token in lowered:
        if token in WH_WORDS:
            return token
    return ""


def capitalized_phrases(text: str) -> list[str]:
    # Exclude the first token of the question as a weak sentence-initial artifact.
    question_words = re.findall(r"\b[A-Za-z][A-Za-z0-9'’-]*\b", text)
    first_word = question_words[0] if question_words else ""
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'’-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'’-]+|[A-Z]{2,}))*"
    )
    phrases = []
    for match in pattern.finditer(text):
        phrase = match.group(0).strip()
        if phrase == first_word and match.start() == 0:
            continue
        if phrase.lower() in {"i"}:
            continue
        phrases.append(normalize_answer(phrase))
    return list(dict.fromkeys(phrases))


def numbers(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b\d+(?:[.,]\d+)?(?:%|°)?\b", text)))


def mean_available(values: list[float]) -> float:
    available = [value for value in values if not pd.isna(value)]
    return sum(available) / len(available) if available else math.nan


def contains_answer(text: str, answer: str) -> bool:
    clean_answer = normalize_answer(answer)
    if not clean_answer:
        return False
    return clean_answer in normalize_answer(text)


def compute_row(row: pd.Series) -> dict[str, object]:
    original_prompt = text_value(row["original_prompt"])
    perturbed_prompt = text_value(row["perturbed_prompt"])
    context = text_value(row["context"])
    original_question = text_value(row["original_question"])
    paraphrased_question = text_value(row["paraphrased_question"])
    reference_answer = text_value(row["reference_answer"])

    full_original_tokens = token_set(original_prompt)
    full_paraphrase_tokens = token_set(perturbed_prompt)
    question_original_tokens = token_set(original_question)
    question_paraphrase_tokens = token_set(paraphrased_question)
    original_content = set(content_tokens(original_question))
    paraphrase_content = set(content_tokens(paraphrased_question))
    context_tokens = token_set(context)
    context_content = set(content_tokens(context))

    original_wh = first_wh_word(original_question)
    paraphrased_wh = first_wh_word(paraphrased_question)
    wh_preserved = (
        math.nan
        if not original_wh
        else int(original_wh == paraphrased_wh)
    )

    original_caps = set(capitalized_phrases(original_question))
    paraphrased_caps = set(capitalized_phrases(paraphrased_question))
    original_numbers = set(numbers(original_question))
    paraphrased_numbers = set(numbers(paraphrased_question))

    question_content_recall = recall(original_content, paraphrase_content)
    cap_recall = recall(original_caps, paraphrased_caps)
    number_recall = recall(original_numbers, paraphrased_numbers)
    critical_cue_preservation = mean_available(
        [question_content_recall, cap_recall, number_recall, wh_preserved]
    )
    cue_disruption = (
        math.nan if pd.isna(critical_cue_preservation) else 1 - critical_cue_preservation
    )

    metrics = row.to_dict()
    metrics.update(
        {
            "full_prompt_token_jaccard": jaccard(full_original_tokens, full_paraphrase_tokens),
            "full_prompt_token_recall": recall(full_original_tokens, full_paraphrase_tokens),
            "question_token_jaccard": jaccard(question_original_tokens, question_paraphrase_tokens),
            "question_token_recall": recall(question_original_tokens, question_paraphrase_tokens),
            "question_content_jaccard": jaccard(original_content, paraphrase_content),
            "question_content_recall": question_content_recall,
            "original_wh_word": original_wh,
            "paraphrased_wh_word": paraphrased_wh,
            "wh_word_preserved": wh_preserved,
            "capitalized_phrase_recall": cap_recall,
            "number_recall": number_recall,
            "reference_answer_in_original_question": contains_answer(original_question, reference_answer),
            "reference_answer_in_paraphrased_question": contains_answer(paraphrased_question, reference_answer),
            "reference_answer_in_context": contains_answer(context, reference_answer),
            "critical_cue_preservation": critical_cue_preservation,
            "cue_disruption": cue_disruption,
            "original_question_context_jaccard": jaccard(question_original_tokens, context_tokens),
            "paraphrased_question_context_jaccard": jaccard(question_paraphrase_tokens, context_tokens),
            "question_context_overlap_delta": jaccard(question_paraphrase_tokens, context_tokens)
            - jaccard(question_original_tokens, context_tokens),
            "original_question_context_content_jaccard": jaccard(original_content, context_content),
            "paraphrased_question_context_content_jaccard": jaccard(paraphrase_content, context_content),
            "question_context_content_overlap_delta": jaccard(paraphrase_content, context_content)
            - jaccard(original_content, context_content),
            "question_context_content_overlap_loss": -(
                jaccard(paraphrase_content, context_content)
                - jaccard(original_content, context_content)
            ),
            "n_capitalized_phrases": len(original_caps),
            "n_numbers": len(original_numbers),
            "n_question_content_tokens": len(original_content),
            "question_length_tokens": len(tokens(original_question)),
            "context_length_tokens": len(tokens(context)),
            "capitalized_phrases_original": "; ".join(sorted(original_caps)),
            "capitalized_phrases_paraphrased": "; ".join(sorted(paraphrased_caps)),
            "numbers_original": "; ".join(sorted(original_numbers)),
            "numbers_paraphrased": "; ".join(sorted(paraphrased_numbers)),
        }
    )
    return metrics


def validate_metrics(branch: str, df: pd.DataFrame) -> None:
    failures = []
    if len(df) != 50:
        failures.append(f"expected 50 rows, found {len(df)}")
    if df["item_id"].nunique() != 50:
        failures.append(f"expected 50 unique item_id values, found {df['item_id'].nunique()}")
    if df["critical_cue_preservation"].isna().any():
        failures.append("critical_cue_preservation contains NA values")
    if not df["critical_cue_preservation"].between(0, 1).all():
        failures.append("critical_cue_preservation has values outside [0, 1]")
    if not df["cue_disruption"].between(0, 1).all():
        failures.append("cue_disruption has values outside [0, 1]")
    if int(df["context_changed"].sum()) != 0:
        failures.append("context_changed is not zero in fixed cue metrics input")
    if failures:
        raise SystemExit(f"{branch}: validation failed: " + "; ".join(failures))


def run_branch(branch: str) -> pd.DataFrame:
    config = BRANCHES[branch]
    input_path = config["input"]
    output_path = config["output"]
    if not input_path.exists():
        raise SystemExit(f"Missing input table: {input_path}")
    table = pd.read_csv(input_path)
    metrics = pd.DataFrame([compute_row(row) for _, row in table.iterrows()])
    validate_metrics(branch, metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(
        f"{branch}: rows={len(metrics)}, "
        f"mean critical_cue_preservation={metrics['critical_cue_preservation'].mean():.6f}, "
        f"mean cue_disruption={metrics['cue_disruption'].mean():.6f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=["outputs", "qwen", "llama", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    for branch in branches:
        run_branch(branch)


if __name__ == "__main__":
    main()
