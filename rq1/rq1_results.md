# RQ1 Results

## Current Status

This document records the current results for RQ1. The main reported similarity metric is Sentence-BERT cosine similarity using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

RQ1 asks:

```text
After applying a sampling-noise correction, is the ranking of the five perturbation types by their effect on output semantic similarity consistent across factual QA, math reasoning, code generation, and open-ended writing?
```

Current formal pipeline status:

| Step | Status |
|---|---|
| Formal original prompt sample | Complete |
| Formal original output generation | Complete |
| Formal sampling-noise baseline | Complete |
| Formal perturbed prompt file | Complete |
| Formal perturbed output generation | Complete |
| Formal noise-corrected perturbation analysis | Complete |

## Formal RQ1a Baseline Results

### Setup

```text
model = gpt-4o-mini
provider = OpenAI API
task_types = factual_qa, math_reasoning, code_generation, open_ended_writing
prompts_per_task_type = 10
total_original_prompts = 40
generations_per_prompt = 5
total_original_generations = 200
temperature = 0.7
top_p = 0.9
max_output_tokens = not set
similarity_metric = Sentence-BERT cosine similarity
embedding_model = sentence-transformers/all-MiniLM-L6-v2
```

Dataset sources:

| Task type | Dataset |
|---|---|
| factual_qa | SQuAD V2 |
| math_reasoning | MATH / Hendrycks MATH |
| code_generation | HumanEvalPack Python |
| open_ended_writing | Alpaca |

### Calculation

For each original prompt, five outputs were generated under the same model and generation settings. The five outputs produce ten pairwise comparisons:

```text
number_of_pairs = 5 choose 2 = 10
mean_within_prompt_similarity = average pairwise similarity among the five outputs
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

The task-level baseline is the average prompt-level sampling-noise drift within each task type.

### Task-Level Baseline

| Task type | Number of prompts | Mean within-prompt similarity | Mean sampling-noise drift | Std. sampling-noise drift |
|---|---:|---:|---:|---:|
| code_generation | 10 | 0.931653 | 0.068347 | 0.028244 |
| factual_qa | 10 | 0.951691 | 0.048309 | 0.064513 |
| math_reasoning | 10 | 0.913485 | 0.086515 | 0.044816 |
| open_ended_writing | 10 | 0.921510 | 0.078490 | 0.072555 |

### Interpretation

In the formal baseline run, `math_reasoning` shows the highest average sampling-noise drift, followed by `open_ended_writing`, `code_generation`, and `factual_qa`.

This means that, before introducing perturbations, repeated outputs for math reasoning prompts varied the most semantically in this sample, while factual QA outputs were the most stable on average. This baseline is important because later perturbation effects should be interpreted relative to these task-specific natural variation levels.

## Formal RQ1b Perturbation Results

The formal perturbed prompt file has been generated:

```text
prompts/rq1_formal_perturbed_prompts.csv
```

It contains:

```text
40 original prompts x 5 perturbation types = 200 perturbed prompts
```

Perturbation counts:

| Perturbation type | Number of perturbed prompts |
|---|---:|
| paraphrasing | 40 |
| reordering | 40 |
| formatting_changes | 40 |
| context_injection | 40 |
| surface_noise | 40 |

The formal perturbed output generation has been completed:

```text
200 perturbed prompts x 5 generations = 1000 perturbed outputs
```

After that, the formal RQ1b analysis will calculate:

```text
baseline_similarity = mean similarity among repeated original outputs
perturbation_similarity = mean cross-similarity between original outputs and perturbed outputs
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

### Noise-Corrected Drift Summary

| Task type | Perturbation type | n items | Mean noise-corrected drift | Std. drift |
|---|---|---:|---:|---:|
| code_generation | context_injection | 10 | 0.002075 | 0.010662 |
| code_generation | formatting_changes | 10 | 0.004808 | 0.016313 |
| code_generation | paraphrasing | 10 | 0.034560 | 0.046164 |
| code_generation | reordering | 10 | 0.007690 | 0.020209 |
| code_generation | surface_noise | 10 | 0.001728 | 0.014665 |
| factual_qa | context_injection | 10 | -0.005502 | 0.016638 |
| factual_qa | formatting_changes | 10 | 0.026844 | 0.065635 |
| factual_qa | paraphrasing | 10 | 0.112586 | 0.099281 |
| factual_qa | reordering | 10 | 0.041203 | 0.060820 |
| factual_qa | surface_noise | 10 | -0.006236 | 0.019244 |
| math_reasoning | context_injection | 10 | 0.000390 | 0.027616 |
| math_reasoning | formatting_changes | 10 | -0.001141 | 0.023136 |
| math_reasoning | paraphrasing | 10 | 0.005822 | 0.015136 |
| math_reasoning | reordering | 10 | -0.000518 | 0.024330 |
| math_reasoning | surface_noise | 10 | -0.006036 | 0.021087 |
| open_ended_writing | context_injection | 10 | 0.018683 | 0.038177 |
| open_ended_writing | formatting_changes | 10 | 0.002976 | 0.016395 |
| open_ended_writing | paraphrasing | 10 | 0.175192 | 0.214344 |
| open_ended_writing | reordering | 10 | 0.035788 | 0.123109 |
| open_ended_writing | surface_noise | 10 | 0.002013 | 0.016223 |

### Heatmap View

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.034560 | 0.112586 | 0.005822 | 0.175192 |
| reordering | 0.007690 | 0.041203 | -0.000518 | 0.035788 |
| formatting_changes | 0.004808 | 0.026844 | -0.001141 | 0.002976 |
| context_injection | 0.002075 | -0.005502 | 0.000390 | 0.018683 |
| surface_noise | 0.001728 | -0.006236 | -0.006036 | 0.002013 |

### Before vs After Baseline Correction

The uncorrected heatmap uses:

```text
uncorrected_drift = 1 - perturbation_similarity
```

The corrected heatmap uses:

```text
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

Uncorrected drift before baseline correction:

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.102908 | 0.160896 | 0.092337 | 0.253682 |
| reordering | 0.076037 | 0.089512 | 0.085997 | 0.114277 |
| formatting_changes | 0.073155 | 0.075153 | 0.085373 | 0.081465 |
| context_injection | 0.070422 | 0.042807 | 0.086905 | 0.097173 |
| surface_noise | 0.070075 | 0.042073 | 0.080479 | 0.080503 |

After baseline correction, many perturbation effects shrink toward zero. This shows why the sampling-noise baseline matters: the uncorrected heatmap includes both perturbation-induced drift and ordinary output variability from repeated LLM sampling.

Generated figures:

```text
figures/rq1_uncorrected_drift_heatmap.png
figures/rq1_noise_corrected_drift_heatmap.png
figures/rq1_uncorrected_vs_corrected_heatmaps.png
```

### Interpretation

The formal RQ1b results suggest that perturbation effects are task-dependent. `Paraphrasing` produced the largest positive noise-corrected drift for `open_ended_writing`, `factual_qa`, and `code_generation`. For `math_reasoning`, all perturbation types were close to zero, suggesting that the perturbed outputs were not much less semantically similar than ordinary repeated outputs from the same original prompt.

`Reordering` had a moderate positive effect for `factual_qa` and `open_ended_writing`, but only a small effect for `code_generation` and near-zero effect for `math_reasoning`. `Surface_noise` and `context_injection` generally produced small or negative drift values, meaning their effects were close to or below the ordinary sampling-noise baseline in this run.

The strongest observed drift was:

```text
open_ended_writing + paraphrasing = 0.175192
factual_qa + paraphrasing = 0.112586
code_generation + paraphrasing = 0.034560
math_reasoning + paraphrasing = 0.005822
```

This pattern supports the idea that the same perturbation type does not affect all task types equally.

## Historical Pilot Notes

Earlier RQ1 pilot outputs used a smaller sample:

```text
5 prompts per task type
20 original prompts total
3 generations per prompt
60 original outputs total
```

Those pilot results were useful for testing the pipeline but should no longer be treated as the main RQ1 result. The current formal baseline above replaces the earlier pilot baseline for reporting purposes.

The baseline stability calibration is still useful as methodological support:

| Task type | n=3 | n=5 | n=7 | n=10 |
|---|---:|---:|---:|---:|
| code_generation | 0.073214 | 0.065846 | 0.062641 | 0.058804 |
| factual_qa | 0.041099 | 0.035543 | 0.031909 | 0.031575 |
| math_reasoning | 0.077373 | 0.063475 | 0.061063 | 0.060591 |
| open_ended_writing | 0.178248 | 0.169233 | 0.165099 | 0.168513 |

The calibration supported using five generations per prompt for the formal run because five outputs provide ten pairwise comparisons per prompt while keeping API cost manageable.

## Related Files

Formal prompt and output files:

```text
prompts/rq1_sampled_original_prompts.csv
prompts/rq1_formal_perturbed_prompts.csv
outputs/rq1_formal_original_generations.csv
outputs/rq1_formal_perturbed_generations.csv
outputs/sbert_rq1_formal_baseline_by_item.csv
outputs/sbert_rq1_formal_baseline_by_task.csv
outputs/sbert_rq1_formal_perturbation_effects_by_item.csv
outputs/sbert_rq1_formal_perturbation_summary.csv
outputs/sbert_rq1_formal_heatmap_noise_corrected_drift.csv
outputs/sbert_rq1_formal_uncorrected_perturbation_summary.csv
outputs/sbert_rq1_formal_uncorrected_heatmap_drift.csv
outputs/sbert_rq1_formal_corrected_heatmap_drift.csv
figures/rq1_uncorrected_drift_heatmap.png
figures/rq1_noise_corrected_drift_heatmap.png
figures/rq1_uncorrected_vs_corrected_heatmaps.png
```

Formal scripts:

```text
src/06_sample_benchmark_prompts.py
src/07_generate_rq1_outputs_openai.py
src/11_generate_rq1b_perturbed_outputs_openai.py
src/19_create_rq1_perturbed_prompts.py
src/22_analyze_rq1_formal_baseline_sbert.py
src/23_analyze_rq1_formal_perturbations_sbert.py
rq1_visualization_code.md
```

Historical pilot files:

```text
outputs/rq1_generations.csv
outputs/sbert_rq1a_noise_by_task.csv
outputs/rq1_calibration_generations.csv
outputs/sbert_rq1_baseline_stability_by_task.csv
outputs/rq1b_pilot_perturbed_generations.csv
outputs/sbert_rq1b_perturbation_summary.csv
```
