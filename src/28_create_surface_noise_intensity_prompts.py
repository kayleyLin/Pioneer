"""Create surface-noise perturbation prompts at multiple intensity levels.

This script does not call any API. It creates three versions of each formal RQ1
original prompt:

    surface_noise_low
    surface_noise_medium
    surface_noise_high

The intensity levels are operationalized by the number of eligible word tokens
that receive a small spelling error. This is intended as an exploratory
"dose-response" analysis for RQ1.
"""

import csv
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "prompts" / "rq1_sampled_original_prompts.csv"
DEFAULT_OUTPUT = ROOT / "prompts" / "rq1_surface_noise_intensity_prompts.csv"
SEED = 20260702

LEVELS = [
    {
        "perturbation_type": "surface_noise_low",
        "perturbation_intensity": "low",
        "minimum_errors": 1,
        "token_ratio": 0.0,
        "description": "one eligible word receives one spelling error",
    },
    {
        "perturbation_type": "surface_noise_medium",
        "perturbation_intensity": "medium",
        "minimum_errors": 2,
        "token_ratio": 0.03,
        "description": "approximately 3% of eligible words receive spelling errors",
    },
    {
        "perturbation_type": "surface_noise_high",
        "perturbation_intensity": "high",
        "minimum_errors": 3,
        "token_ratio": 0.08,
        "description": "approximately 8% of eligible words receive spelling errors",
    },
]

MAX_ERRORS_PER_PROMPT = 12

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

EXCLUDED_WORDS = {
    "true",
    "false",
    "none",
    "null",
    "input",
    "output",
    "return",
    "class",
    "def",
    "self",
    "context",
    "question",
    "problem",
    "instruction",
}


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
        "perturbation_intensity",
        "surface_noise_error_count",
        "surface_noise_eligible_token_count",
        "surface_noise_token_ratio",
        "construction_method",
        "method_reference",
        "semantic_equivalence_status",
        "original_prompt",
        "perturbed_prompt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def eligible_token_matches(prompt: str) -> list[re.Match[str]]:
    matches = list(re.finditer(r"\b[A-Za-z]{4,}\b", prompt))
    eligible = []
    for match in matches:
        token = match.group(0)
        if token.lower() in EXCLUDED_WORDS:
            continue
        if any(char.isdigit() for char in token) or "_" in token:
            continue
        eligible.append(match)
    return eligible


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


def error_count(eligible_count: int, minimum_errors: int, token_ratio: float) -> int:
    if eligible_count == 0:
        return 0
    count = max(minimum_errors, math.ceil(eligible_count * token_ratio))
    return min(count, eligible_count, MAX_ERRORS_PER_PROMPT)


def add_surface_noise(prompt: str, count: int, rng: random.Random) -> str:
    matches = eligible_token_matches(prompt)
    if not matches or count <= 0:
        return prompt

    selected = sorted(rng.sample(matches, k=min(count, len(matches))), key=lambda m: m.start())
    pieces = []
    cursor = 0
    for match in selected:
        token = match.group(0)
        pieces.append(prompt[cursor : match.start()])
        pieces.append(apply_spelling_error(token, rng))
        cursor = match.end()
    pieces.append(prompt[cursor:])
    return "".join(pieces)


def main() -> None:
    prompts = read_csv(DEFAULT_INPUT)
    rows: list[dict[str, str]] = []

    for prompt_row in prompts:
        prompt = prompt_row["prompt_text"]
        eligible_count = len(eligible_token_matches(prompt))

        for level in LEVELS:
            rng = random.Random(
                f"{SEED}:{prompt_row['item_id']}:{level['perturbation_type']}"
            )
            count = error_count(
                eligible_count,
                int(level["minimum_errors"]),
                float(level["token_ratio"]),
            )
            perturbed_prompt = add_surface_noise(prompt, count, rng)
            rows.append(
                {
                    "item_id": prompt_row["item_id"],
                    "task_type": prompt_row["task_type"],
                    "dataset_name": prompt_row["dataset_name"],
                    "source_index": prompt_row["source_index"],
                    "perturbation_type": level["perturbation_type"],
                    "perturbation_intensity": level["perturbation_intensity"],
                    "surface_noise_error_count": str(count),
                    "surface_noise_eligible_token_count": str(eligible_count),
                    "surface_noise_token_ratio": str(level["token_ratio"]),
                    "construction_method": "rule_based_posix_spelling_error_intensity",
                    "method_reference": (
                        "Reference 3 POSIX spelling-error operations; "
                        "Reference 1 PromptRobust character-level perturbations"
                    ),
                    "semantic_equivalence_status": "pending_manual_check",
                    "original_prompt": prompt,
                    "perturbed_prompt": perturbed_prompt,
                }
            )

    write_csv(DEFAULT_OUTPUT, rows)
    print(f"Wrote {len(rows)} surface-noise intensity prompts to {DEFAULT_OUTPUT}")
    for level in LEVELS:
        n_rows = sum(row["perturbation_type"] == level["perturbation_type"] for row in rows)
        print(f"{level['perturbation_type']}: {n_rows} rows")


if __name__ == "__main__":
    main()
