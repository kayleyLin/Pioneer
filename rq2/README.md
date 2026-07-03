# RQ2 Workspace

This folder contains the working materials for RQ2.

RQ2 asks whether, for tasks with an objective correctness criterion, semantic drift is associated with a change in task correctness.

Current RQ2 scope:

| Included task type | Dataset | Correctness basis |
|---|---|---|
| factual_qa | SQuAD V2 | Reference-answer matching |
| math_reasoning | MATH / Hendrycks MATH | Final-answer equivalence |
| code_generation | HumanEvalPack Python | Functional test execution |

Open-ended writing is excluded because it does not have an objective correctness criterion.

## Files

| File | Purpose |
|---|---|
| `rq2_methodology_design.md` | RQ2-specific methodology design |
| `data_dictionary.md` | Data schemas for RQ2 intermediate and final files |
| `correctness_evaluator_notes.md` | Rules used by the correctness evaluator |
| `outputs/` | RQ2-specific generated outputs |

RQ1 methodology source:

```text
../pioneer methodology.md
```

## Main Script

The current evaluator script is:

```bash
/opt/anaconda3/bin/python src/23_evaluate_rq2_correctness.py
```

Default input:

```text
outputs/rq1_formal_original_generations.csv
prompts/rq1_sampled_original_prompts.csv
```

Default output:

```text
rq2/outputs/rq2_original_correctness_by_generation.csv
rq2/outputs/rq2_original_correctness_summary_by_task.csv
```
