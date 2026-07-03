# RQ2 Data Dictionary

## Correctness By Generation

Planned file:

```text
rq2/outputs/rq2_original_correctness_by_generation.csv
```

Later perturbed-output file:

```text
rq2/outputs/rq2_perturbed_correctness_by_generation.csv
```

Columns:

| Column | Meaning |
|---|---|
| `item_id` | Stable item identifier from the RQ1 sample |
| `task_type` | `factual_qa`, `math_reasoning`, or `code_generation` |
| `dataset_name` | Source benchmark dataset |
| `source_index` | Source row index in the dataset |
| `source_id` | Source dataset item id, when available |
| `sample_id` | Repeated generation index |
| `model_name` | Output-generation model |
| `prompt_variant` | `original` or `perturbed` |
| `perturbation_type` | Perturbation type; blank for original prompts |
| `reference_answer` | Dataset reference answer or canonical solution |
| `output_text` | LLM output being evaluated |
| `extracted_answer` | Extracted answer/code used for automatic evaluation |
| `performance_score` | Task-specific automatic performance score; factual QA uses containment-first token F1, math/code use 1/0 |
| `factual_containment_match` | For factual QA only: whether the normalized reference answer appears in the normalized output |
| `factual_token_f1` | For factual QA only: SQuAD-style token F1 between the normalized reference answer and full output |
| `is_correct` | `true` or `false` for math/code; blank for factual QA because factual QA uses continuous `performance_score` |
| `needs_manual_review` | `true` when automatic evaluation is uncertain |
| `correctness_method` | Method used for the decision |
| `notes` | Short explanation or error message |

## Correctness Summary By Task

Planned file:

```text
rq2/outputs/rq2_original_correctness_summary_by_task.csv
```

Columns:

| Column | Meaning |
|---|---|
| `task_type` | Task type |
| `n_outputs` | Number of evaluated outputs |
| `n_correct` | Number automatically labeled correct |
| `n_incorrect` | Number automatically labeled incorrect |
| `n_unlabeled_correctness` | Number without a binary correctness label, mainly factual QA rows |
| `n_manual_review` | Number flagged for manual review |
| `correct_rate_auto_only` | Correct rate among rows with binary labels and no manual review |
| `mean_performance_score` | Mean task-specific performance score among rows not requiring manual review |

## Future Drift-Correctness Analysis Table

Planned file:

```text
rq2/outputs/rq2_drift_correctness_analysis_by_item.csv
```

Columns:

| Column | Meaning |
|---|---|
| `item_id` | Stable item identifier |
| `task_type` | Included task type |
| `perturbation_type` | Prompt perturbation type |
| `semantic_drift` | Noise-corrected semantic drift from RQ1 analysis |
| `original_correct_rate` | Correctness rate across repeated original generations |
| `perturbed_correct_rate` | Correctness rate across repeated perturbed generations |
| `correctness_change_magnitude` | Absolute change in correctness rate |
| `correctness_changed` | Binary indicator for whether correctness rate changed |
