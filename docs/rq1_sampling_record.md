# RQ1 Sampling Record

## Sampling Settings

```text
random_seed = 20260623
sample_size_per_task_type = 10
total_sampled_prompts = 40
```

## Task Types And Dataset Sources

| Task type | Dataset | Config | Split |
|---|---|---|---|
| factual_qa | rajpurkar/squad_v2 | squad_v2 | validation |
| math_reasoning | nlile/hendrycks-MATH-benchmark | default | train |
| code_generation | bigcode/humanevalpack | python | test |
| open_ended_writing | tatsu-lab/alpaca | default | train |

## Current Sample Counts

| Task type | Number of prompts |
|---|---:|
| factual_qa | 10 |
| math_reasoning | 10 |
| code_generation | 10 |
| open_ended_writing | 10 |

## Recorded Fields

The sampled prompt file records the following information for each prompt:

```text
item_id
task_type
dataset_name
dataset_split
source_index
source_id
prompt_text
reference_answer
random_seed
```

These fields are saved in:

```text
prompts/rq1_sampled_original_prompts.csv
```

## Methodology Wording Draft

Possible wording for the methodology section:

```text
For the formal RQ1 experiment, I used stratified random sampling by task type. Four task types were included: factual question answering, mathematical reasoning, code generation, and open-ended writing. For stronger literature alignment, factual QA was sampled from SQuAD V2 and mathematical reasoning was sampled from MATH / Hendrycks MATH. For each task type, ten prompts were randomly selected from a corresponding public benchmark dataset using a fixed random seed of 20260623. The fixed seed was used to make the sampling procedure reproducible. For each sampled item, I recorded the task type, dataset name, dataset split, source index, prompt text, reference answer, and random seed.
```
