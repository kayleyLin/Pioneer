# RQ1a Baseline Results

## Purpose

RQ1a estimates the sampling-noise baseline before introducing prompt perturbations.

Current RQ1a question:

```text
How much sampling noise occurs when the same prompt is repeatedly given to the same LLM, and does this sampling-noise baseline differ across task types?
```

This is the first component of the full RQ1. The full RQ1 still requires perturbation analysis after the baseline is established.

## Experimental Setup

Prompt sampling:

```text
task_types = factual_qa, math_reasoning, code_generation, open_ended_writing
sample_size_per_task_type = 5
total_prompts = 20
sampling_method = stratified random sampling by task type
random_seed = 20260623
```

Generation settings:

```text
model = gpt-4o-mini
n_samples_per_prompt = 3
temperature = 0.7
top_p = 0.9
max_output_tokens = 300
total_generations = 60
similarity_metric = Sentence-BERT cosine similarity
embedding_model = sentence-transformers/all-MiniLM-L6-v2
```

## Calculation

For each prompt, the exact same prompt was submitted to the same model three times. The three outputs were compared pairwise:

```text
similarity(output_1, output_2)
similarity(output_1, output_3)
similarity(output_2, output_3)
```

Then:

```text
mean_within_prompt_similarity = average(pairwise similarities)
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

The task-level baseline was calculated by averaging sampling_noise_drift across the five prompts in each task type.

## Task-Level Results

| Task type | Number of prompts | Mean within-prompt similarity | Mean sampling-noise drift | Std. sampling-noise drift |
|---|---:|---:|---:|---:|
| factual_qa | 5 | 0.949619 | 0.050381 | 0.049847 |
| math_reasoning | 5 | 0.939735 | 0.060265 | 0.031659 |
| code_generation | 5 | 0.942352 | 0.057648 | 0.026224 |
| open_ended_writing | 5 | 0.909664 | 0.090336 | 0.086416 |

## Preliminary Interpretation

The Sentence-BERT pilot suggests that open-ended writing has the highest sampling-noise drift. Factual QA, code generation, and math reasoning have lower drift values.

This pattern is consistent with the idea that tasks with larger output spaces produce more natural variation across repeated generations.

## Limitations

These results are preliminary.

Current limitations:

```text
Only 5 prompts per task type.
Only 3 generations per prompt.
Only one commercial API model.
```

Formal analysis should increase the sample size if time and resources allow.

## Related Files

```text
prompts/rq1_sampled_original_prompts.csv
outputs/rq1_generations.csv
outputs/rq1_real_noise_by_item.csv
outputs/rq1_real_noise_by_task.csv
outputs/sbert_rq1a_noise_by_item.csv
outputs/sbert_rq1a_noise_by_task.csv
src/09_analyze_rq1_real_generations.py
src/16_analyze_rq1_with_sentence_bert.py
```
