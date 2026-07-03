# Pioneer Research Project

This repository contains code, prompts, analysis scripts, and research notes for a Pioneer research project on task-dependent LLM output variability and prompt perturbation sensitivity.

## Project Focus

The current active research question is RQ1:

```text
Are LLM output noise baselines and perturbation effects task-dependent across different task types?
```

The project studies four task types:

- factual QA
- math reasoning
- code generation
- open-ended writing

The main analysis uses repeated LLM generations, prompt perturbations, Sentence-BERT similarity, sampling-noise correction, and statistical tests.

## Repository Structure

```text
src/        Python scripts for sampling, generation, perturbation, and analysis
scripts/    Batch scripts for longer RQ1 runs
config/     Generation configuration
docs/       Methodology and sampling notes
design/     Literature-aligned design notes
rq1/        RQ1-specific methodology and result notes
rq2/        RQ2 exploratory notes and scripts
outputs/    Generated experiment outputs and analysis tables
prompts/    Sampled and perturbed prompt files
figures/    Generated figures
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For OpenAI API calls, set the API key locally before running generation scripts:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Do not commit API keys or `.env` files.

## Notes

Large personal files, PDFs, Word documents, screenshots, local virtual environments, and logs are excluded from version control through `.gitignore`.
