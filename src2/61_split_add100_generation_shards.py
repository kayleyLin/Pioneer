"""Split add100 original and perturbed prompt CSVs into generation shards."""

from __future__ import annotations

import argparse
import csv
import re
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
FACTUAL_REWRITE_RE = re.compile(
    r"\b(rewrite|paraphrase)\b.*\b(prompt|question)\b|^\s*Rewrite\s*:",
    flags=re.IGNORECASE | re.DOTALL,
)
MATH_ARTIFACT_RE = re.compile(
    r"Research Question\s*:|Rewrite\s*:|Rewrite the prompt|"
    r"Code Signature\s*:|Code signature\s*:|Task Signature\s*:|```|"
    r"def\s+\w+\s*\(|function\s+\w+\s*\(",
    flags=re.IGNORECASE,
)


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists() or (path.parts and path.parts[0].lower() == "pioneer"):
        return path.resolve()
    return ROOT / path


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


def load_many(paths: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        rows, current_fields = read_csv(resolve(path))
        if fieldnames is None:
            fieldnames = current_fields
        elif fieldnames != current_fields:
            raise SystemExit(f"Fieldnames differ in {path}")
        all_rows.extend(rows)
    return all_rows, fieldnames or []


def validate_original(rows: list[dict[str, str]], expected_items_per_task: int) -> None:
    if not rows:
        raise SystemExit("No original prompt rows loaded")
    bad_tasks = sorted({row["task_type"] for row in rows} - TASKS)
    if bad_tasks:
        raise SystemExit(f"Unexpected original prompt task_type values: {bad_tasks}")
    for task in sorted(TASKS):
        count = sum(row["task_type"] == task for row in rows)
        if count != expected_items_per_task:
            raise SystemExit(
                f"{task}: expected {expected_items_per_task} original rows, got {count}"
            )
    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise SystemExit("Duplicate original item_id values found")


def validate_perturbed(rows: list[dict[str, str]], expected_items_per_task: int) -> None:
    if not rows:
        raise SystemExit("No perturbed prompt rows loaded")
    bad_tasks = sorted({row["task_type"] for row in rows} - TASKS)
    if bad_tasks:
        raise SystemExit(f"Unexpected perturbed prompt task_type values: {bad_tasks}")
    bad_perturbations = sorted(
        {row["perturbation_type"] for row in rows} - PERTURBATIONS
    )
    if bad_perturbations:
        raise SystemExit(f"Unexpected perturbation_type values: {bad_perturbations}")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["item_id"], row["perturbation_type"])
        if key in seen:
            raise SystemExit(f"Duplicate perturbed prompt key found: {key}")
        seen.add(key)

    perturbations_by_item: dict[str, set[str]] = {}
    for row in rows:
        perturbations_by_item.setdefault(row["item_id"], set()).add(
            row["perturbation_type"]
        )
    incomplete = {
        item_id: sorted(perturbations)
        for item_id, perturbations in perturbations_by_item.items()
        if perturbations != PERTURBATIONS
    }
    if incomplete:
        examples = list(incomplete.items())[:10]
        raise SystemExit(f"Items missing perturbations: {examples}")

    for task in sorted(TASKS):
        task_rows = [row for row in rows if row["task_type"] == task]
        expected = expected_items_per_task * len(PERTURBATIONS)
        if len(task_rows) != expected:
            raise SystemExit(f"{task}: expected {expected} perturbed rows, got {len(task_rows)}")

    validate_factual_paraphrases(rows)
    validate_math_paraphrases(rows)


def validate_factual_paraphrases(rows: list[dict[str, str]]) -> None:
    bad_residue: list[str] = []
    bad_shape: list[str] = []
    for row in rows:
        if row["task_type"] != "factual_qa" or row["perturbation_type"] != "paraphrasing":
            continue
        prompt = row.get("perturbed_prompt", "")
        if FACTUAL_REWRITE_RE.search(prompt):
            bad_residue.append(row["item_id"])
        if not re.search(r"(?is)^\s*Context\s*:.+\n\s*\n\s*Question\s*:.+", prompt):
            bad_shape.append(row["item_id"])
    if bad_residue:
        raise SystemExit(
            "factual_qa paraphrasing prompts still contain rewrite/paraphrase residue: "
            f"{bad_residue[:10]}"
        )
    if bad_shape:
        raise SystemExit(
            "factual_qa paraphrasing prompts must be clean Context + Question prompts: "
            f"{bad_shape[:10]}"
        )


def validate_math_paraphrases(rows: list[dict[str, str]]) -> None:
    bad_artifact: list[str] = []
    bad_asy: list[str] = []
    for row in rows:
        if row["task_type"] != "math_reasoning" or row["perturbation_type"] != "paraphrasing":
            continue
        prompt = row.get("perturbed_prompt", "")
        if MATH_ARTIFACT_RE.search(prompt):
            bad_artifact.append(row["item_id"])
        if "[asy]" in row.get("original_prompt", "") and "[asy]" not in prompt:
            bad_asy.append(row["item_id"])
    if bad_artifact:
        raise SystemExit(
            "math_reasoning paraphrasing prompts still contain template artifacts: "
            f"{bad_artifact[:10]}"
        )
    if bad_asy:
        raise SystemExit(
            "math_reasoning paraphrasing prompts removed original [asy] diagram blocks: "
            f"{bad_asy[:10]}"
        )


def split_round_robin(rows: list[dict[str, str]], shards: int) -> list[list[dict[str, str]]]:
    out = [[] for _ in range(shards)]
    for index, row in enumerate(rows):
        out[index % shards].append(row)
    return out


def write_manifest(
    path: Path,
    original_shards: list[list[dict[str, str]]],
    perturbed_shards: list[list[dict[str, str]]],
) -> None:
    rows = []
    for index, (original_rows, perturbed_rows) in enumerate(
        zip(original_shards, perturbed_shards),
        start=1,
    ):
        rows.append(
            {
                "shard": index,
                "original_prompt_rows": len(original_rows),
                "perturbed_prompt_rows": len(perturbed_rows),
                "expected_original_generation_rows": len(original_rows) * 5,
                "expected_perturbed_generation_rows": len(perturbed_rows) * 5,
            }
        )
    write_csv(path, rows, list(rows[0].keys()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-inputs",
        nargs="+",
        type=Path,
        default=[Path("prompts/rq1_sampled_original_prompts_add100_three_task.csv")],
    )
    parser.add_argument(
        "--perturbed-inputs",
        nargs="+",
        type=Path,
        default=[Path("prompts/rq1_formal_perturbed_prompts_add100_three_task.csv")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tmp_parallel/add100"))
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--expected-items-per-task", type=int, default=100)
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")

    original_rows, original_fields = load_many(args.original_inputs)
    perturbed_rows, perturbed_fields = load_many(args.perturbed_inputs)
    validate_original(original_rows, args.expected_items_per_task)
    validate_perturbed(perturbed_rows, args.expected_items_per_task)

    original_shards = split_round_robin(original_rows, args.shards)
    perturbed_shards = split_round_robin(perturbed_rows, args.shards)
    output_dir = resolve(args.output_dir)

    for index, rows in enumerate(original_shards, start=1):
        path = output_dir / f"original_prompts_shard{index}.csv"
        write_csv(path, rows, original_fields)
        print(f"Wrote {len(rows)} original prompt rows: {path}")

    for index, rows in enumerate(perturbed_shards, start=1):
        path = output_dir / f"perturbed_prompts_shard{index}.csv"
        write_csv(path, rows, perturbed_fields)
        print(f"Wrote {len(rows)} perturbed prompt rows: {path}")

    manifest = output_dir / "manifest.csv"
    write_manifest(manifest, original_shards, perturbed_shards)
    print(f"Wrote shard manifest: {manifest}")


if __name__ == "__main__":
    main()
