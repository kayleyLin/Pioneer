# RQ1 Sampling Algorithm

## Purpose

The goal of this step is to select a small but balanced pilot sample of original prompts for RQ1.

RQ1 measures sampling noise only, so the sampled prompts are not perturbed at this stage.

## Sampling Method

The sampling method is stratified random sampling by task type.

This means the project does not randomly sample 20 prompts from one combined pool. Instead, it samples the same number of prompts from each task type.

Current pilot setting:

```text
task types = 4
sample size per task type = 5
total prompts = 20
random seed = 20260623
```

## Algorithm

For each task type:

```text
1. Select the benchmark dataset assigned to that task type.
2. Get the total number of available rows in that dataset split.
3. Use a fixed random seed to randomly choose 5 row indices.
4. Fetch those rows from the dataset.
5. Extract the prompt text and reference answer.
6. Save the sampled prompt and its source metadata to CSV.
```

In simplified Python-style pseudocode:

```python
rng = random.Random(20260623)

for task_type in task_types:
    dataset = dataset_for_task_type[task_type]
    total_rows = get_total_rows(dataset)
    sampled_indices = rng.sample(range(total_rows), 5)

    for index in sampled_indices:
        row = get_dataset_row(dataset, index)
        save_prompt_with_metadata(row)
```

## Why This Is Reproducible

The fixed random seed means the random sampling process can be repeated. If the same script is run with the same dataset versions and seed, it should select the same row indices.

## Current Implementation

The sampling script is:

```text
src/06_sample_benchmark_prompts.py
```

The sampled output file is:

```text
prompts/rq1_sampled_original_prompts.csv
```

