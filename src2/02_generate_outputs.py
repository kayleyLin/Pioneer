"""Generate repeated pilot outputs for original and perturbed prompts.

This first version uses a local mock generator so the RQ1 analysis pipeline can
be tested before connecting a real LLM API.
"""

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROMPTS = ROOT / "prompts" / "original_prompts.csv"
PERTURBED_PROMPTS = ROOT / "prompts" / "perturbed_prompts.csv"
GENERATIONS = ROOT / "outputs" / "generations.csv"


N_SAMPLES = 3
MODEL_NAME = "pilot_mock"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def stable_choice(options: list[str], key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(options)
    return options[index]


def mock_generate(row: dict[str, str], sample_id: int) -> str:
    item_id = row["item_id"]
    task_type = row["task_type"]
    perturbation_type = row["perturbation_type"]
    key = f"{item_id}:{perturbation_type}:{sample_id}"

    if item_id == "fact_001":
        options = [
            "The capital city of Canada is Ottawa.",
            "Canada's capital is Ottawa.",
            "Ottawa is the capital of Canada.",
        ]
    elif item_id == "fact_002":
        options = [
            "Pride and Prejudice was written by Jane Austen.",
            "The author is Jane Austen.",
            "Jane Austen wrote the novel Pride and Prejudice.",
        ]
    elif item_id == "math_001":
        options = [
            "Maya spends 21 dollars in total.",
            "The total cost is 7 times 3, which equals 21 dollars.",
            "She spends $21.",
        ]
    elif item_id == "math_002":
        options = [
            "The train travels 150 miles.",
            "Distance equals speed times time, so 60 x 2.5 = 150 miles.",
            "It travels 150 miles in total.",
        ]
    elif item_id == "code_001":
        options = [
            "def add_numbers(a, b):\n    return a + b",
            "def add_numbers(x, y):\n    return x + y",
            "def add_numbers(a, b):\n    result = a + b\n    return result",
        ]
    elif item_id == "write_001":
        options = [
            "Sleep is important for students because it helps memory, focus, and emotional health. When students get enough rest, they can learn more effectively and manage stress better.",
            "Students need enough sleep to stay alert in class, remember what they study, and maintain their health. Poor sleep can make schoolwork harder and reduce concentration.",
            "Getting enough sleep supports students' learning, mood, and physical well-being. It gives the brain time to recover and prepare for the next day.",
        ]
    else:
        options = [f"Mock output for {task_type}."]

    if perturbation_type == "surface_noise" and task_type in {"math_reasoning", "code_generation"}:
        options = options + [
            "There may be a typo in the question, but the answer should be checked carefully.",
        ]
    if perturbation_type == "context_injection" and task_type == "open_ended_writing":
        options = options + [
            "In a school wellness discussion, sleep matters because it affects attention, memory, stress, and daily energy for students.",
        ]

    return stable_choice(options, key)


def main() -> None:
    original_rows = read_csv(ORIGINAL_PROMPTS)
    perturbed_rows = read_csv(PERTURBED_PROMPTS)

    rows: list[dict[str, str | int]] = []

    for prompt_row in original_rows:
        row = {
            "item_id": prompt_row["item_id"],
            "task_type": prompt_row["task_type"],
            "perturbation_type": "none",
            "prompt_text": prompt_row["original_prompt"],
        }
        for sample_id in range(1, N_SAMPLES + 1):
            rows.append(
                {
                    "item_id": row["item_id"],
                    "task_type": row["task_type"],
                    "prompt_version": "original",
                    "perturbation_type": "none",
                    "sample_id": sample_id,
                    "model_name": MODEL_NAME,
                    "output_text": mock_generate(row, sample_id),
                }
            )

    for prompt_row in perturbed_rows:
        for sample_id in range(1, N_SAMPLES + 1):
            rows.append(
                {
                    "item_id": prompt_row["item_id"],
                    "task_type": prompt_row["task_type"],
                    "prompt_version": "perturbed",
                    "perturbation_type": prompt_row["perturbation_type"],
                    "sample_id": sample_id,
                    "model_name": MODEL_NAME,
                    "output_text": mock_generate(prompt_row, sample_id),
                }
            )

    with GENERATIONS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "item_id",
                "task_type",
                "prompt_version",
                "perturbation_type",
                "sample_id",
                "model_name",
                "output_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} generations to {GENERATIONS}")


if __name__ == "__main__":
    main()
