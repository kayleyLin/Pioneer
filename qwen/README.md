# Qwen n=50 and n=150 Experiments

This directory contains generation outputs and analysis results using **qwen-plus** (DashScope API) instead of gpt-4o-mini.

## Setup

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

## Generation Scripts

| Script | Purpose |
|---|---|
| `src/07_generate_rq1_outputs_qwen.py` | Generate original outputs from sampled prompts |
| `src/11_generate_rq1b_perturbed_outputs_qwen.py` | Generate perturbed outputs from perturbed prompts |

## Batch Run

```bash
# Bash:
bash scripts/run_qwen_n50.sh

# PowerShell:
$env:DASHSCOPE_API_KEY = "your-key"
pwsh -ExecutionPolicy Bypass -File scripts/run_qwen_n50.ps1
```

## Output Files

```
qwen/outputs/
├── rq1_qwen_original_generations_n50_factual_qa.csv
├── rq1_qwen_original_generations_n50_math_reasoning.csv
├── rq1_qwen_original_generations_n50_code_generation.csv
├── rq1_qwen_original_generations_n50_open_ended_writing.csv
├── rq1_qwen_perturbed_generations_n50_factual_qa.csv
├── rq1_qwen_perturbed_generations_n50_math_reasoning.csv
├── rq1_qwen_perturbed_generations_n50_code_generation.csv
└── rq1_qwen_perturbed_generations_n50_open_ended_writing.csv
```

## Analysis

After generation, run the existing analysis scripts with paths updated:

```bash
# Baseline analysis (modify GENERATION_FILES paths in a copy of 31)
# Perturbation analysis (modify GENERATION_FILES paths in a copy of 32)
# Heatmaps (modify CORRECTED/UNCORRECTED paths in a copy of 34)
```

## Three-task add100 / n=150 run

The 2026-07-14 add100 run adds 100 factual QA, 100 math reasoning, and 100 code generation items. It keeps add100 outputs separate and then builds a validated three-task n=150 corpus.

Key audited files:

```text
qwen/add100_run_manifest.json
qwen/outputs/add100_generation_validation_report.md
qwen/outputs/rq1_qwen_original_generations_add100_three_task.csv
qwen/outputs/rq1_qwen_perturbed_generations_add100_three_task.csv
qwen/outputs/n150_three_task_merge_validation_report.md
qwen/outputs/rq1_qwen_original_generations_n150_three_task.csv
qwen/outputs/rq1_qwen_perturbed_generations_n150_three_task_fixed.csv
qwen/outputs/rq1_qwen_prompts_n150_three_task.csv
qwen/outputs/sbert_rq1_n150_three_task_baseline_by_item.csv
qwen/outputs/sbert_rq1_n150_three_task_perturbation_effects_by_item.csv
qwen/rq2_outputs_n150/
```

The generation runner supports smoke, original, perturbed, and all phases, uses isolated shards, and resumes only missing generation keys. The SBERT and correctness scripts accept input/output parameters; n=50 defaults remain available.
