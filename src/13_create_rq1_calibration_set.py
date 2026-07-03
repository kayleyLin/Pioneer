"""Create a small calibration set for repeated-generation count.

The calibration set uses two prompts per task type from the existing RQ1
sampled prompt file.
"""

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "prompts" / "rq1_sampled_original_prompts.csv"
OUTPUT = ROOT / "prompts" / "rq1_calibration_prompts.csv"
K_PER_TASK = 2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    rows = read_csv(INPUT)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)

    selected: list[dict[str, str]] = []
    for task_type in sorted(grouped):
        task_rows = grouped[task_type][:K_PER_TASK]
        if len(task_rows) < K_PER_TASK:
            raise ValueError(f"Not enough prompts for {task_type}")
        selected.extend(task_rows)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} calibration prompts to {OUTPUT}")


if __name__ == "__main__":
    main()
