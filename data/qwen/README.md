# Qwen n=50 Experiment

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
