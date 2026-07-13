"""Prepare prompt subsets for paraphrasing generation repair."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paraphrase_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows = read_csv(path)
    selected = [row for row in rows if row["perturbation_type"] == "paraphrasing"]
    if not selected:
        raise SystemExit(f"No paraphrasing rows found in {path}")
    return selected, list(rows[0].keys())


def split_rows(rows: list[dict[str, str]], shards: int) -> list[list[dict[str, str]]]:
    buckets = [[] for _ in range(shards)]
    for index, row in enumerate(rows):
        buckets[index % shards].append(row)
    return buckets


def main() -> None:
    out_dir = ROOT / "outputs" / "paraphrase_repair_prompts"
    shard_dir = out_dir / "shards"

    add100_factual, fields = paraphrase_rows(
        ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_factual_qa.csv"
    )
    add100_math, _ = paraphrase_rows(
        ROOT / "prompts" / "rq1_formal_perturbed_prompts_add100_math_reasoning.csv"
    )
    n50_math_rows, _ = paraphrase_rows(
        ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv"
    )
    n50_math_5201 = [
        row for row in n50_math_rows if row["item_id"] == "math_reasoning_5201"
    ]
    if len(n50_math_5201) != 1:
        raise SystemExit(
            "Expected exactly one n50 math paraphrasing prompt for math_reasoning_5201, "
            f"found {len(n50_math_5201)}"
        )

    combined = []
    for row in add100_factual:
        tagged = dict(row)
        tagged["repair_scope"] = "add100_factual_paraphrasing"
        combined.append(tagged)
    for row in add100_math:
        tagged = dict(row)
        tagged["repair_scope"] = "add100_math_paraphrasing"
        combined.append(tagged)

    combined_fields = fields + ["repair_scope"]
    write_csv(out_dir / "add100_factual_paraphrasing_prompts.csv", add100_factual, fields)
    write_csv(out_dir / "add100_math_paraphrasing_prompts.csv", add100_math, fields)
    write_csv(out_dir / "n50_math_5201_paraphrasing_prompt.csv", n50_math_5201, fields)
    write_csv(out_dir / "add100_combined_paraphrasing_prompts.csv", combined, combined_fields)

    for index, bucket in enumerate(split_rows(combined, 5), start=1):
        write_csv(shard_dir / f"add100_paraphrasing_repair_shard{index}.csv", bucket, combined_fields)

    print(f"add100 factual paraphrasing prompts: {len(add100_factual)}")
    print(f"add100 math paraphrasing prompts: {len(add100_math)}")
    print("n50 math repair prompts: 1")
    print(f"wrote repair prompts to: {out_dir}")


if __name__ == "__main__":
    main()
