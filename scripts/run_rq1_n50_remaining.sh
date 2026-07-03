#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/wenfenglin/Desktop/Pioneer"
cd "$ROOT"

mkdir -p logs outputs prompts
LOG="logs/rq1_n50_remaining_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Run: export OPENAI_API_KEY=\"your_api_key_here\""
  exit 1
fi

run_step() {
  local name="$1"
  shift
  echo ""
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') START: ${name} =====" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') END: ${name} =====" | tee -a "$LOG"
}

run_step "prepare task-specific n50 prompt files" python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

root = Path("/Users/wenfenglin/Desktop/Pioneer")
source = root / "prompts" / "rq1_sampled_original_prompts_n50.csv"
tasks = [
    "factual_qa",
    "math_reasoning",
    "code_generation",
    "open_ended_writing",
]

rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
for task in tasks:
    subset = [row for row in rows if row["task_type"] == task]
    target = root / "prompts" / f"rq1_sampled_original_prompts_n50_{task}.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(subset)
    print(
        f"{task}: wrote {len(subset)} prompts to {target}; "
        f"datasets={dict(Counter(row['dataset_name'] for row in subset))}; "
        f"empty_reference_answers={sum(not row.get('reference_answer', '').strip() for row in subset)}"
    )
PY

run_step "baseline original outputs: code_generation" \
  python src/07_generate_rq1_outputs_openai.py \
    --input prompts/rq1_sampled_original_prompts_n50_code_generation.csv \
    --output outputs/rq1_formal_original_generations_n50_code_generation.csv

run_step "baseline original outputs: open_ended_writing" \
  python src/07_generate_rq1_outputs_openai.py \
    --input prompts/rq1_sampled_original_prompts_n50_open_ended_writing.csv \
    --output outputs/rq1_formal_original_generations_n50_open_ended_writing.csv

for task in math_reasoning code_generation open_ended_writing; do
  run_step "formal perturbed prompts: ${task}" \
    python src/19_create_rq1_perturbed_prompts.py \
      --input "prompts/rq1_sampled_original_prompts_n50_${task}.csv" \
      --output "prompts/rq1_formal_perturbed_prompts_n50_${task}.csv"
done

for task in factual_qa math_reasoning code_generation open_ended_writing; do
  run_step "perturbed outputs: ${task}" \
    env \
      RQ1B_PERTURBED_PROMPTS="prompts/rq1_formal_perturbed_prompts_n50_${task}.csv" \
      RQ1B_PERTURBED_OUTPUTS="outputs/rq1_formal_perturbed_generations_n50_${task}.csv" \
      python src/11_generate_rq1b_perturbed_outputs_openai.py
done

run_step "final row-count summary" python - <<'PY'
import csv
from pathlib import Path

paths = [
    "outputs/rq1_formal_original_generations_n50_factual_qa.csv",
    "outputs/rq1_formal_original_generations_n50_math_reasoning.csv",
    "outputs/rq1_formal_original_generations_n50_code_generation.csv",
    "outputs/rq1_formal_original_generations_n50_open_ended_writing.csv",
    "prompts/rq1_formal_perturbed_prompts_n50_factual_qa.csv",
    "prompts/rq1_formal_perturbed_prompts_n50_math_reasoning.csv",
    "prompts/rq1_formal_perturbed_prompts_n50_code_generation.csv",
    "prompts/rq1_formal_perturbed_prompts_n50_open_ended_writing.csv",
    "outputs/rq1_formal_perturbed_generations_n50_factual_qa.csv",
    "outputs/rq1_formal_perturbed_generations_n50_math_reasoning.csv",
    "outputs/rq1_formal_perturbed_generations_n50_code_generation.csv",
    "outputs/rq1_formal_perturbed_generations_n50_open_ended_writing.csv",
]

root = Path("/Users/wenfenglin/Desktop/Pioneer")
for relative in paths:
    path = root / relative
    if not path.exists():
        print(f"{relative}: MISSING")
        continue
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    print(f"{relative}: {len(rows)} rows")
PY

echo ""
echo "All queued RQ1 n50 remaining steps completed. Log: ${LOG}"
