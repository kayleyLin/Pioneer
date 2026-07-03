"""Validate prompt files for the pilot experiment."""

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROMPTS = ROOT / "prompts" / "original_prompts.csv"
PERTURBED_PROMPTS = ROOT / "prompts" / "perturbed_prompts.csv"
EXPECTED_PERTURBATIONS = {
    "paraphrasing",
    "reordering",
    "formatting",
    "context_injection",
    "surface_noise",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    original_rows = read_csv(ORIGINAL_PROMPTS)
    perturbed_rows = read_csv(PERTURBED_PROMPTS)

    original_ids = {row["item_id"] for row in original_rows}
    perturbation_counts = Counter(row["item_id"] for row in perturbed_rows)

    print(f"Original prompts: {len(original_rows)}")
    print(f"Perturbed prompts: {len(perturbed_rows)}")

    missing_items = original_ids - set(perturbation_counts)
    if missing_items:
        raise ValueError(f"Missing perturbations for items: {sorted(missing_items)}")

    for item_id in sorted(original_ids):
        item_rows = [row for row in perturbed_rows if row["item_id"] == item_id]
        found_types = {row["perturbation_type"] for row in item_rows}
        missing_types = EXPECTED_PERTURBATIONS - found_types
        extra_types = found_types - EXPECTED_PERTURBATIONS

        if missing_types or extra_types:
            raise ValueError(
                f"{item_id} has missing types {sorted(missing_types)} "
                f"and extra types {sorted(extra_types)}"
            )

        print(f"{item_id}: {perturbation_counts[item_id]} perturbations")

    print("Prompt pilot set is ready.")


if __name__ == "__main__":
    main()
