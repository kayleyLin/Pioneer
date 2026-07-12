"""Run Qwen generation scripts with 5-way parallelism.

Splits each task's input CSV into 5 chunks, runs 5 parallel instances,
then merges results into one output file.

Usage:
    python src/run_qwen_parallel.py --step 07
    python src/run_qwen_parallel.py --step 11
    python src/run_qwen_parallel.py --step all

Requires DASHSCOPE_API_KEY environment variable.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rq1_generation_config.json"
N_PARALLEL = 5
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

TASKS = ["factual_qa", "math_reasoning", "code_generation", "open_ended_writing"]


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def configured_sample_count() -> int:
    return int(json.loads(CONFIG.read_text(encoding="utf-8"))["n_samples_per_prompt"])


def generation_key(row: dict[str, str]) -> tuple[str, ...]:
    if "perturbation_type" in row:
        return (
            row.get("item_id", ""),
            row.get("perturbation_type", ""),
            row.get("sample_id", ""),
        )
    return (row.get("item_id", ""), row.get("sample_id", ""))


def prompt_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("item_id", ""), row.get("perturbation_type", ""))


def split_csv(input_path: Path, n_chunks: int):
    """Split CSV into n_chunks by rows. Returns list of temp file paths."""
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    chunk_size = len(rows) // n_chunks
    remainder = len(rows) % n_chunks
    chunks = []
    temp_files = []

    for i in range(n_chunks):
        start = i * chunk_size + min(i, remainder)
        end = (i + 1) * chunk_size + min(i + 1, remainder)
        chunk_rows = rows[start:end]
        if not chunk_rows:
            continue

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False,
            encoding="utf-8", newline="",
            dir=ROOT / "tmp_parallel",
        )
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunk_rows)
        tmp.close()
        chunks.append(Path(tmp.name))
        temp_files.append(Path(tmp.name))
        print(f"  Chunk {i+1}: {len(chunk_rows)} rows -> {tmp.name}")

    return chunks, fieldnames, temp_files


def run_single(script: str, input_path: Path, output_path: Path, chunk_label: str = ""):
    """Run one instance of a generation script."""
    cmd = [
        sys.executable, str(ROOT / "src" / script),
        "--input", str(input_path),
        "--output", str(output_path),
    ]
    env = os.environ.copy()
    env["DASHSCOPE_API_KEY"] = API_KEY
    label = f"[{chunk_label}] " if chunk_label else ""
    print(f"  {label}Starting: {' '.join(cmd)}", flush=True)
    log_file = output_path.with_suffix(".log")
    with open(log_file, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p, log_file


def run_single_11(script: str, input_path: Path, output_path: Path, chunk_label: str = ""):
    """Run one instance of script 11 using env vars."""
    cmd = [sys.executable, str(ROOT / "src" / script)]
    env = os.environ.copy()
    env["DASHSCOPE_API_KEY"] = API_KEY
    env["RQ1B_QWEN_PERTURBED_PROMPTS"] = str(input_path)
    env["RQ1B_QWEN_PERTURBED_OUTPUTS"] = str(output_path)
    label = f"[{chunk_label}] " if chunk_label else ""
    print(f"  {label}Starting: {' '.join(cmd)}", flush=True)
    log_file = output_path.with_suffix(".log")
    with open(log_file, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\nENVS: API_KEY={'set' if API_KEY else 'NOT SET'}\n")
        f.flush()
        p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p, log_file


def merge_csvs(output_files: list[Path], final_path: Path, fieldnames):
    """Merge multiple CSV outputs into one, deduplicating by key."""
    seen = set()
    merged = []
    output_fieldnames = None

    for f in output_files:
        if not f.exists():
            print(f"  Warning: {f} does not exist, skipping")
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if output_fieldnames is None:
                output_fieldnames = reader.fieldnames
            for row in reader:
                if "perturbation_type" in row:
                    key = tuple(
                        row.get(k, "")
                        for k in ["item_id", "perturbation_type", "sample_id"]
                    )
                else:
                    key = (row.get("item_id", ""), row.get("sample_id", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(row)

    if output_fieldnames is None:
        output_fieldnames = fieldnames

    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    print(f"  Merged {len(merged)} rows -> {final_path}")


def seed_chunk_outputs_from_final(
    chunks: list[Path],
    chunk_outputs: list[Path],
    final_path: Path,
) -> None:
    """Copy rows from an incomplete final output back into per-chunk outputs."""
    if not final_path.exists():
        return

    with open(final_path, newline="", encoding="utf-8") as fh:
        final_reader = csv.DictReader(fh)
        final_fieldnames = final_reader.fieldnames
        final_rows = list(final_reader)
    if not final_fieldnames or not final_rows:
        return

    final_by_prompt: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in final_rows:
        final_by_prompt.setdefault(prompt_key(row), []).append(row)

    for chunk, chunk_output in zip(chunks, chunk_outputs):
        chunk_prompt_keys = {prompt_key(row) for row in read_csv(chunk)}
        rows_by_key: dict[tuple[str, ...], dict[str, str]] = {}

        if chunk_output.exists():
            with open(chunk_output, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows_by_key[generation_key(row)] = row

        for key in chunk_prompt_keys:
            for row in final_by_prompt.get(key, []):
                rows_by_key.setdefault(generation_key(row), row)

        if not rows_by_key:
            continue

        chunk_output.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=final_fieldnames)
            writer.writeheader()
            writer.writerows(rows_by_key.values())
        print(
            f"  Seeded {len(rows_by_key)} existing rows -> {chunk_output}",
            flush=True,
        )


def run_step_07(task: str):
    """Run code 07 with 5-way parallelism for one task."""
    print(f"\n=== Code 07: {task} (5-way parallel) ===", flush=True)
    input_path = ROOT / "prompts" / f"rq1_sampled_original_prompts_n50_{task}.csv"
    final_output = ROOT / "qwen" / "outputs" / f"rq1_qwen_original_generations_n50_{task}.csv"

    print(f"  Input path: {input_path}, exists: {input_path.exists()}", flush=True)
    if not input_path.exists():
        print(f"  Input not found: {input_path}", flush=True)
        return

    # Split input
    tmp_dir = ROOT / "tmp_parallel"
    tmp_dir.mkdir(exist_ok=True)

    chunks, fieldnames, temp_files = split_csv(input_path, N_PARALLEL)

    # Create chunk-specific output files
    chunk_outputs = []
    for i, chunk in enumerate(chunks):
        chunk_out = tmp_dir / f"output_07_{task}_chunk{i+1}.csv"
        chunk_outputs.append(chunk_out)

    # Launch 5 parallel processes
    processes = []
    log_files = []
    for i, (chunk, chunk_out) in enumerate(zip(chunks, chunk_outputs)):
        p, log = run_single("07_generate_rq1_outputs_qwen.py", chunk, chunk_out, f"07-{task}-{i+1}")
        processes.append(p)
        log_files.append(log)
        print(f"  [07-{task}-{i+1}] Log: {log}", flush=True)

    # Wait for all to finish
    print(f"  Waiting for {len(processes)} processes...", flush=True)
    for p in processes:
        p.wait()

    # Print summary
    print(f"\n  --- Chunk Results for {task} ---", flush=True)
    for i, log in enumerate(log_files):
        if log.exists():
            lines = log.read_text().strip().split("\n")
            last = lines[-3:] if len(lines) >= 3 else lines
            print(f"  Chunk {i+1} last lines: {' | '.join(last)}", flush=True)
        if chunk_outputs[i].exists():
            with open(chunk_outputs[i], newline="", encoding="utf-8") as fh:
                n = sum(1 for _ in csv.DictReader(fh))
            print(f"  Chunk {i+1}: {n} rows", flush=True)

    # Merge
    merge_csvs(chunk_outputs, final_output, fieldnames)

    # Cleanup temp files (keep logs for debugging)
    for f in temp_files + chunk_outputs:
        if f.exists():
            f.unlink()
    print(f"  Done! Output: {final_output}", flush=True)


def run_step_11(task: str):
    """Run code 11 with 5-way parallelism for one task."""
    print(f"\n=== Code 11: {task} (5-way parallel) ===")
    input_path = ROOT / "prompts" / f"rq1_formal_perturbed_prompts_n50_{task}.csv"
    final_output = ROOT / "qwen" / "outputs" / f"rq1_qwen_perturbed_generations_n50_{task}.csv"

    if not input_path.exists():
        print(f"  Input not found: {input_path}")
        return

    expected_rows = csv_row_count(input_path) * configured_sample_count()
    existing_final_rows = csv_row_count(final_output)
    if existing_final_rows >= expected_rows:
        print(
            f"  Final output already complete: {existing_final_rows}/"
            f"{expected_rows} rows -> {final_output}"
        )
        return

    tmp_dir = ROOT / "tmp_parallel"
    tmp_dir.mkdir(exist_ok=True)

    chunks, fieldnames, temp_files = split_csv(input_path, N_PARALLEL)

    chunk_outputs = []
    for i, chunk in enumerate(chunks):
        chunk_out = tmp_dir / f"output_11_{task}_chunk{i+1}.csv"
        chunk_outputs.append(chunk_out)

    processes = []
    log_files = []
    for i, (chunk, chunk_out) in enumerate(zip(chunks, chunk_outputs)):
        p, log = run_single_11("11_generate_rq1b_perturbed_outputs_qwen.py", chunk, chunk_out, f"11-{task}-{i+1}")
        processes.append(p)
        log_files.append(log)
        print(f"  [11-{task}-{i+1}] Log: {log}", flush=True)

    print(f"  Waiting for {len(processes)} processes...", flush=True)
    for p in processes:
        p.wait()

    print(f"\n  --- Chunk Results for {task} ---", flush=True)
    for i, log in enumerate(log_files):
        if log.exists():
            lines = log.read_text().strip().split("\n")
            last = lines[-3:] if len(lines) >= 3 else lines
            print(f"  Chunk {i+1} last lines: {' | '.join(last)}", flush=True)
        if chunk_outputs[i].exists():
            with open(chunk_outputs[i], newline="", encoding="utf-8") as fh:
                n = sum(1 for _ in csv.DictReader(fh))
            print(f"  Chunk {i+1}: {n} rows", flush=True)

    merge_csvs(chunk_outputs, final_output, fieldnames)

    for f in temp_files + chunk_outputs:
        if f.exists():
            f.unlink()
    print(f"  Done! Output: {final_output}", flush=True)


def main():
    print(f"API_KEY set: {bool(API_KEY)}, args: {sys.argv}", flush=True)
    if not API_KEY:
        print("Error: DASHSCOPE_API_KEY not set.")
        print("Run: $env:DASHSCOPE_API_KEY = 'your-key'")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python run_qwen_parallel.py --step 07|11|all [--tasks task1,task2,...]")
        sys.exit(1)

    # Parse args
    step = None
    tasks_to_run = TASKS
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--step" and i + 1 < len(sys.argv):
            step = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--tasks" and i + 1 < len(sys.argv):
            tasks_to_run = sys.argv[i + 1].split(",")
            i += 2
        else:
            # positional arg as step
            if step is None:
                step = sys.argv[i]
            i += 1

    if not step:
        print("Usage: python run_qwen_parallel.py --step 07|11|all")
        sys.exit(1)

    print(f"Step: {step}, Tasks: {tasks_to_run}", flush=True)

    if step in ("07", "all"):
        for task in tasks_to_run:
            run_step_07(task)

    if step in ("11", "all"):
        for task in tasks_to_run:
            run_step_11(task)

    print("\n=== All parallel generation complete ===")


if __name__ == "__main__":
    main()
