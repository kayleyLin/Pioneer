"""Create RQ1 perturbed prompts using the finalized perturbation methods.

Before running the full version:

    export OPENAI_API_KEY="your_api_key_here"
    python src/19_create_rq1_perturbed_prompts.py

For a non-API smoke test:

    python src/19_create_rq1_perturbed_prompts.py --dry-run-no-api
"""

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
DEFAULT_INPUT = ROOT / "prompts" / "rq1_sampled_original_prompts.csv"
DEFAULT_OUTPUT = ROOT / "prompts" / "rq1_formal_perturbed_prompts.csv"
SEED = 20260623

PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]

NEUTRAL_CONTEXT_SENTENCES = [
    "This prompt is part of a general language-model evaluation.",
    "The wording below is being used in a controlled prompt study.",
    "This item appears in a broader collection of benchmark prompts.",
    "The following request is provided for a routine model-response comparison.",
]

KEYBOARD_NEIGHBORS = {
    "a": "s",
    "b": "n",
    "c": "v",
    "d": "s",
    "e": "r",
    "f": "g",
    "g": "h",
    "h": "j",
    "i": "o",
    "j": "k",
    "k": "l",
    "l": "k",
    "m": "n",
    "n": "m",
    "o": "p",
    "p": "o",
    "q": "w",
    "r": "t",
    "s": "a",
    "t": "y",
    "u": "i",
    "v": "b",
    "w": "e",
    "x": "z",
    "y": "u",
    "z": "x",
}


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "item_id",
        "task_type",
        "dataset_name",
        "source_index",
        "perturbation_type",
        "construction_method",
        "method_reference",
        "semantic_equivalence_status",
        "original_prompt",
        "perturbed_prompt",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def paraphrase_prompt(prompt: str, config: dict, dry_run_no_api: bool) -> str:
    if dry_run_no_api:
        return prompt

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Run:\n"
            'export OPENAI_API_KEY="your_api_key_here"\n'
            "or use --dry-run-no-api for a non-API smoke test."
        )

    client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
    model = config["paraphrase_generation_model"]
    response = client.chat.completions.create(
        model=model,
        temperature=config["temperature"],
        top_p=config["top_p"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite prompts for a research study. Preserve the exact "
                    "task intent, answer target, constraints, numbers, entities, "
                    "examples, and code signatures. Return only the rewritten prompt."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Paraphrase the following prompt while preserving its meaning:\n\n"
                    f"{prompt}"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def reorder_prompt(prompt: str) -> str:
    context_question = re.match(
        r"^Context:\s*(.+?)\s+Question:\s*(.+)$", prompt.strip(), flags=re.DOTALL
    )
    if context_question:
        context = context_question.group(1).strip()
        question = context_question.group(2).strip()
        return f"Question: {question}\n\nContext: {context}"

    problem_instruction = re.match(
        r"^Problem:\s*(.+?)\s+Instruction:\s*(.+)$", prompt.strip(), flags=re.DOTALL
    )
    if problem_instruction:
        problem = problem_instruction.group(1).strip()
        instruction = problem_instruction.group(2).strip()
        return f"Instruction: {instruction}\n\nProblem: {problem}"

    instruction_input = re.match(
        r"^Instruction:\s*(.+?)\s+Input:\s*(.+)$", prompt.strip(), flags=re.DOTALL
    )
    if instruction_input:
        instruction = instruction_input.group(1).strip()
        input_text = instruction_input.group(2).strip()
        return f"Input: {input_text}\n\nInstruction: {instruction}"

    if "\n\nInput:" in prompt:
        instruction, input_text = prompt.split("\n\nInput:", 1)
        return f"Input:{input_text.strip()}\n\nInstruction: {instruction.strip()}"

    if "\nExample\n" in prompt:
        task_text, example_text = prompt.split("\nExample\n", 1)
        return f"Example\n{example_text.strip()}\n\nTask:\n{task_text.strip()}"

    doctest_match = re.search(r"\n\s*>>>", prompt)
    if doctest_match:
        task_text = prompt[: doctest_match.start()].strip()
        example_text = prompt[doctest_match.start() :].strip()
        return f"Examples:\n{example_text}\n\nTask:\n{task_text}"

    examples_match = re.search(r"\n\s*Examples\s*\n", prompt, flags=re.IGNORECASE)
    if examples_match:
        task_text = prompt[: examples_match.start()].strip()
        example_text = prompt[examples_match.end() :].strip()
        return f"Examples:\n{example_text}\n\nTask:\n{task_text}"

    problem_marker = " to solve the following problem:"
    if problem_marker in prompt:
        signature_text, problem_text = prompt.split(problem_marker, 1)
        return (
            f"Problem:{problem_text.strip()}\n\n"
            f"Function requirement: {signature_text.strip()}."
        )

    sentences = sentence_split(prompt)
    if len(sentences) >= 2:
        return " ".join([sentences[-1], *sentences[:-1]])

    return prompt


def format_prompt(prompt: str) -> str:
    return f"Please answer the following prompt:\n\n- Prompt: {prompt}"


def inject_context(prompt: str, rng: random.Random) -> str:
    context = rng.choice(NEUTRAL_CONTEXT_SENTENCES)
    return f"{context}\n\n{prompt}"


def eligible_noise_tokens(prompt: str) -> list[str]:
    tokens = re.findall(r"\b[A-Za-z]{4,}\b", prompt)
    return [
        token
        for token in tokens
        if not any(char.isdigit() for char in token)
        and "_" not in token
        and token.lower() not in {"true", "false", "none", "null"}
    ]


def apply_spelling_error(token: str, rng: random.Random) -> str:
    operation = rng.choice(["insertion", "omission", "transposition", "substitution"])
    index = rng.randrange(len(token))

    if operation == "insertion":
        return token[:index] + rng.choice("abcdefghijklmnopqrstuvwxyz") + token[index:]

    if operation == "omission" and len(token) > 1:
        return token[:index] + token[index + 1 :]

    if operation == "transposition" and len(token) > 1:
        index = min(index, len(token) - 2)
        return token[:index] + token[index + 1] + token[index] + token[index + 2 :]

    char = token[index]
    replacement = KEYBOARD_NEIGHBORS.get(char.lower(), rng.choice("abcdefghijklmnopqrstuvwxyz"))
    if char.isupper():
        replacement = replacement.upper()
    return token[:index] + replacement + token[index + 1 :]


def add_surface_noise(prompt: str, rng: random.Random) -> str:
    candidates = eligible_noise_tokens(prompt)
    if not candidates:
        return prompt

    token = rng.choice(candidates)
    noisy_token = apply_spelling_error(token, rng)
    return re.sub(rf"\b{re.escape(token)}\b", noisy_token, prompt, count=1)


def method_details(perturbation_type: str) -> tuple[str, str]:
    details = {
        "paraphrasing": (
            "llm_assisted_posix_paraphrase",
            "Reference 3 POSIX: GPT-3.5-Turbo paraphrase generation",
        ),
        "reordering": (
            "rule_based_prompt_component_reordering",
            "Reference 5 Haase et al.: information-order variation",
        ),
        "formatting_changes": (
            "rule_based_template_transformation",
            "Reference 3 POSIX: prompt-template variation",
        ),
        "context_injection": (
            "fixed_neutral_sentence_bank",
            "Reference 1 PromptRobust: irrelevant sentence insertion, softened",
        ),
        "surface_noise": (
            "rule_based_posix_spelling_error",
            "Reference 3 POSIX: insertion/omission/transposition/substitution",
        ),
    }
    return details[perturbation_type]


def build_perturbation(
    prompt: str, perturbation_type: str, config: dict, rng: random.Random, dry_run_no_api: bool
) -> str:
    if perturbation_type == "paraphrasing":
        return paraphrase_prompt(prompt, config, dry_run_no_api)
    if perturbation_type == "reordering":
        return reorder_prompt(prompt)
    if perturbation_type == "formatting_changes":
        return format_prompt(prompt)
    if perturbation_type == "context_injection":
        return inject_context(prompt, rng)
    if perturbation_type == "surface_noise":
        return add_surface_noise(prompt, rng)
    raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run-no-api",
        action="store_true",
        help="Create non-API perturbations and leave paraphrasing unchanged.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    config = read_config(CONFIG)
    prompts = read_csv(input_path)
    rows: list[dict[str, str]] = []

    for prompt_row in prompts:
        original_prompt = prompt_row["prompt_text"]
        for perturbation_type in PERTURBATION_ORDER:
            print(
                f"Creating {perturbation_type} for {prompt_row['item_id']}",
                flush=True,
            )
            rng = random.Random(f"{SEED}:{prompt_row['item_id']}:{perturbation_type}")
            construction_method, method_reference = method_details(perturbation_type)
            perturbed_prompt = build_perturbation(
                original_prompt,
                perturbation_type,
                config,
                rng,
                args.dry_run_no_api,
            )
            rows.append(
                {
                    "item_id": prompt_row["item_id"],
                    "task_type": prompt_row["task_type"],
                    "dataset_name": prompt_row["dataset_name"],
                    "source_index": prompt_row["source_index"],
                    "perturbation_type": perturbation_type,
                    "construction_method": construction_method,
                    "method_reference": method_reference,
                    "semantic_equivalence_status": "pending_manual_check",
                    "original_prompt": original_prompt,
                    "perturbed_prompt": perturbed_prompt.strip(),
                }
            )

    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} perturbed prompts to {output_path}")
    if args.dry_run_no_api:
        print("Dry run used: paraphrasing rows were left unchanged.")


if __name__ == "__main__":
    main()
