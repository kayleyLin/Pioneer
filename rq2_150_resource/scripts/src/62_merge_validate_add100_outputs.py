"""Merge and validate add100 generation shard outputs."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {"factual_qa", "math_reasoning", "code_generation"}
PERTURBATIONS = {
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
}

ORIGINAL_FIELDS = [
    "item_id",
    "task_type",
    "dataset_name",
    "source_index",
    "sample_id",
    "model_name",
    "temperature",
    "top_p",
    "max_output_tokens",
    "prompt_text",
    "output_text",
]

PERTURBED_FIELDS = [
    "item_id",
    "task_type",
    "dataset_name",
    "source_index",
    "perturbation_type",
    "sample_id",
    "model_name",
    "temperature",
    "top_p",
    "max_output_tokens",
    "original_prompt",
    "perturbed_prompt",
    "output_text",
]


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists() or (path.parts and path.parts[0].lower() == "pioneer"):
        return path.resolve()
    return ROOT / path


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        raw = Path(pattern)
        if raw.is_absolute():
            pattern_text = str(raw)
        elif raw.parts and raw.parts[0].lower() == "pioneer":
            pattern_text = str((Path.cwd() / raw).resolve())
        else:
            pattern_text = str(ROOT / raw)
        matches = [Path(match) for match in glob.glob(pattern_text)]
        if not matches:
            raise SystemExit(f"No files matched pattern: {pattern}")
        paths.extend(matches)
    return sorted(set(paths), key=lambda path: str(path))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_many(paths: list[Path], expected_fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        file_rows, fields = read_csv(path)
        missing = [field for field in expected_fields if field not in fields]
        if missing:
            raise SystemExit(f"{path} is missing fields: {missing}")
        print(f"Loaded {len(file_rows)} rows from {path}")
        rows.extend(file_rows)
    return rows


def validate_no_empty_outputs(rows: list[dict[str, str]], label: str) -> None:
    empty = [row for row in rows if not row.get("output_text", "").strip()]
    if empty:
        examples = [row.get("item_id", "") for row in empty[:10]]
        raise SystemExit(f"{label}: found {len(empty)} empty output_text rows: {examples}")


def validate_original(
    rows: list[dict[str, str]],
    expected_items_per_task: int,
    expected_samples: int,
) -> tuple[set[str], list[str]]:
    validate_no_empty_outputs(rows, "original")
    bad_tasks = sorted({row["task_type"] for row in rows} - TASKS)
    if bad_tasks:
        raise SystemExit(f"original: unexpected task_type values: {bad_tasks}")

    key_counts = Counter((row["item_id"], row["sample_id"]) for row in rows)
    duplicates = [key for key, count in key_counts.items() if count > 1]
    if duplicates:
        raise SystemExit(f"original: duplicate keys: {duplicates[:10]}")

    by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)

    bad_sample_counts = {
        item_id: len(item_rows)
        for item_id, item_rows in by_item.items()
        if len(item_rows) != expected_samples
    }
    if bad_sample_counts:
        raise SystemExit(f"original: bad sample counts: {list(bad_sample_counts.items())[:10]}")

    for task in sorted(TASKS):
        item_count = len(
            {
                row["item_id"]
                for row in rows
                if row["task_type"] == task
            }
        )
        if item_count != expected_items_per_task:
            raise SystemExit(
                f"original {task}: expected {expected_items_per_task} items, got {item_count}"
            )

    item_order = sorted(by_item)
    expected_rows = len(TASKS) * expected_items_per_task * expected_samples
    if len(rows) != expected_rows:
        raise SystemExit(f"original: expected {expected_rows} rows, got {len(rows)}")
    return set(by_item), item_order


def validate_perturbed(
    rows: list[dict[str, str]],
    original_item_ids: set[str],
    expected_items_per_task: int,
    expected_samples: int,
) -> None:
    validate_no_empty_outputs(rows, "perturbed")
    bad_tasks = sorted({row["task_type"] for row in rows} - TASKS)
    if bad_tasks:
        raise SystemExit(f"perturbed: unexpected task_type values: {bad_tasks}")
    bad_perturbations = sorted(
        {row["perturbation_type"] for row in rows} - PERTURBATIONS
    )
    if bad_perturbations:
        raise SystemExit(f"perturbed: unexpected perturbation_type values: {bad_perturbations}")

    key_counts = Counter(
        (row["item_id"], row["perturbation_type"], row["sample_id"])
        for row in rows
    )
    duplicates = [key for key, count in key_counts.items() if count > 1]
    if duplicates:
        raise SystemExit(f"perturbed: duplicate keys: {duplicates[:10]}")

    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[(row["item_id"], row["perturbation_type"])].append(row)

    perturbed_item_ids = {item_id for item_id, _ in by_group}
    if perturbed_item_ids != original_item_ids:
        missing = sorted(original_item_ids - perturbed_item_ids)[:10]
        extra = sorted(perturbed_item_ids - original_item_ids)[:10]
        raise SystemExit(
            f"perturbed item_id set does not match original. missing={missing}; extra={extra}"
        )

    bad_sample_counts = {
        key: len(group_rows)
        for key, group_rows in by_group.items()
        if len(group_rows) != expected_samples
    }
    if bad_sample_counts:
        raise SystemExit(f"perturbed: bad sample counts: {list(bad_sample_counts.items())[:10]}")

    for item_id in original_item_ids:
        item_perturbations = {
            perturbation
            for current_item, perturbation in by_group
            if current_item == item_id
        }
        if item_perturbations != PERTURBATIONS:
            raise SystemExit(
                f"{item_id}: expected perturbations {sorted(PERTURBATIONS)}, "
                f"got {sorted(item_perturbations)}"
            )

    for task in sorted(TASKS):
        item_count = len(
            {
                row["item_id"]
                for row in rows
                if row["task_type"] == task
            }
        )
        if item_count != expected_items_per_task:
            raise SystemExit(
                f"perturbed {task}: expected {expected_items_per_task} items, got {item_count}"
            )

    expected_rows = (
        len(TASKS)
        * expected_items_per_task
        * len(PERTURBATIONS)
        * expected_samples
    )
    if len(rows) != expected_rows:
        raise SystemExit(f"perturbed: expected {expected_rows} rows, got {len(rows)}")


def sort_original(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (row["task_type"], row["item_id"], int(row["sample_id"])))


def sort_perturbed(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row["task_type"],
            row["item_id"],
            row["perturbation_type"],
            int(row["sample_id"]),
        ),
    )


def write_report(
    path: Path,
    original_rows: list[dict[str, str]],
    perturbed_rows: list[dict[str, str]],
    original_paths: list[Path],
    perturbed_paths: list[Path],
    expected_items_per_task: int,
    expected_samples: int,
) -> None:
    lines = [
        "# add100 generation merge validation report",
        "",
        "## Inputs",
        "",
        "Original shard outputs:",
        "",
    ]
    lines.extend(f"- `{path}`" for path in original_paths)
    lines.extend(["", "Perturbed shard outputs:", ""])
    lines.extend(f"- `{path}`" for path in perturbed_paths)
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Tasks: {', '.join(sorted(TASKS))}",
            f"- Expected items per task: {expected_items_per_task}",
            f"- Expected samples per prompt: {expected_samples}",
            f"- Original rows: {len(original_rows)}",
            f"- Perturbed rows: {len(perturbed_rows)}",
            "- Duplicate keys: none",
            "- Empty outputs: none",
            "- Original and perturbed item sets: match",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-patterns",
        nargs="+",
        default=["outputs/add100_shards/original_generations_shard*.csv"],
    )
    parser.add_argument(
        "--perturbed-patterns",
        nargs="+",
        default=["outputs/add100_shards/perturbed_generations_shard*.csv"],
    )
    parser.add_argument(
        "--original-output",
        type=Path,
        default=Path("outputs/rq1_formal_original_generations_add100_three_task.csv"),
    )
    parser.add_argument(
        "--perturbed-output",
        type=Path,
        default=Path("outputs/rq1_formal_perturbed_generations_add100_three_task.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/add100_generation_validation_report.md"),
    )
    parser.add_argument("--expected-items-per-task", type=int, default=100)
    parser.add_argument("--expected-samples", type=int, default=5)
    args = parser.parse_args()

    original_paths = expand_paths(args.original_patterns)
    perturbed_paths = expand_paths(args.perturbed_patterns)
    original_rows = load_many(original_paths, ORIGINAL_FIELDS)
    perturbed_rows = load_many(perturbed_paths, PERTURBED_FIELDS)

    original_item_ids, _ = validate_original(
        original_rows,
        args.expected_items_per_task,
        args.expected_samples,
    )
    validate_perturbed(
        perturbed_rows,
        original_item_ids,
        args.expected_items_per_task,
        args.expected_samples,
    )

    original_output = resolve(args.original_output)
    perturbed_output = resolve(args.perturbed_output)
    report = resolve(args.report)

    write_csv(original_output, sort_original(original_rows), ORIGINAL_FIELDS)
    write_csv(perturbed_output, sort_perturbed(perturbed_rows), PERTURBED_FIELDS)
    write_report(
        report,
        original_rows,
        perturbed_rows,
        original_paths,
        perturbed_paths,
        args.expected_items_per_task,
        args.expected_samples,
    )

    print(f"Wrote merged original outputs: {original_output}")
    print(f"Wrote merged perturbed outputs: {perturbed_output}")
    print(f"Wrote validation report: {report}")


if __name__ == "__main__":
    main()
