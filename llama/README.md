# Llama n=50 Experiment

This directory contains the Llama version of the RQ1 n=50 pipeline.

It is intentionally separate from the existing OpenAI and Qwen outputs:

- `outputs/`: Llama generation and analysis CSV files
- `figures/`: Llama heatmap PNG files
- `src/`: Llama-specific generation and analysis scripts
- `tmp_parallel/`: temporary files for future parallel runs

## API Setup

The generation scripts expect an OpenAI-compatible chat completions endpoint.
This works with hosted Llama providers and local servers such as vLLM.

Required environment variables:

```powershell
$env:LLAMA_API_URL = "http://localhost:8000/v1/chat/completions"
$env:LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

For hosted providers, also set:

```powershell
$env:LLAMA_API_KEY = "your-provider-key"
```

## Generate Original Outputs

```powershell
python llama/src/07_generate_rq1_outputs_llama.py --input prompts/rq1_sampled_original_prompts_n50_factual_qa.csv --output llama/outputs/rq1_llama_original_generations_n50_factual_qa.csv
python llama/src/07_generate_rq1_outputs_llama.py --input prompts/rq1_sampled_original_prompts_n50_math_reasoning.csv --output llama/outputs/rq1_llama_original_generations_n50_math_reasoning.csv
python llama/src/07_generate_rq1_outputs_llama.py --input prompts/rq1_sampled_original_prompts_n50_code_generation.csv --output llama/outputs/rq1_llama_original_generations_n50_code_generation.csv
python llama/src/07_generate_rq1_outputs_llama.py --input prompts/rq1_sampled_original_prompts_n50_open_ended_writing.csv --output llama/outputs/rq1_llama_original_generations_n50_open_ended_writing.csv
```

## Generate Perturbed Outputs

```powershell
python llama/src/11_generate_rq1b_perturbed_outputs_llama.py --input prompts/rq1_formal_perturbed_prompts_n50_factual_qa.csv --output llama/outputs/rq1_llama_perturbed_generations_n50_factual_qa.csv
python llama/src/11_generate_rq1b_perturbed_outputs_llama.py --input prompts/rq1_formal_perturbed_prompts_n50_math_reasoning.csv --output llama/outputs/rq1_llama_perturbed_generations_n50_math_reasoning.csv
python llama/src/11_generate_rq1b_perturbed_outputs_llama.py --input prompts/rq1_formal_perturbed_prompts_n50_code_generation.csv --output llama/outputs/rq1_llama_perturbed_generations_n50_code_generation.csv
python llama/src/11_generate_rq1b_perturbed_outputs_llama.py --input prompts/rq1_formal_perturbed_prompts_n50_open_ended_writing.csv --output llama/outputs/rq1_llama_perturbed_generations_n50_open_ended_writing.csv
```

## Analyze And Plot

```powershell
python llama/src/31_analyze_rq1_n50_baseline_sbert_llama.py
python llama/src/32_analyze_rq1_n50_perturbations_sbert_llama.py
python llama/src/34_create_rq1_n50_heatmaps_llama.py
```

The target figure will be:

```text
llama/figures/rq1_n50_noise_corrected_drift_heatmap.png
```

