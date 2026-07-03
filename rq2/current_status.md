# RQ2 Current Status

Last updated: 2026-07-01

## Completed

Created the RQ2 workspace:

```text
rq2/
rq2/outputs/
```

Created working documentation:

```text
rq2/README.md
rq2/methodology.md
rq2/data_dictionary.md
rq2/correctness_evaluator_notes.md
rq2/current_status.md
```

Created the first correctness evaluator:

```text
src/23_evaluate_rq2_correctness.py
```

Ran the evaluator on the formal original-prompt generations:

```text
outputs/rq1_formal_original_generations.csv
```

Generated:

```text
rq2/outputs/rq2_original_correctness_by_generation.csv
rq2/outputs/rq2_original_correctness_summary_by_task.csv
```

## Current Original-Output Correctness Summary

The evaluator now uses a fully automatic conservative policy. Rows that cannot be matched or compared reliably by the automatic rule are labeled incorrect rather than sent to manual review.

For factual QA, the implemented method is now:

```text
containment first + token F1 backup
```

If the normalized reference answer appears in the normalized output, the factual performance score is 1.0. Otherwise, the evaluator computes SQuAD-style token F1 between the reference answer and the full output. Factual QA no longer receives a binary correctness label; RQ2 uses the continuous factual performance score for this task.

| Task type | n outputs | n correct | n incorrect | n unlabeled correctness | n manual review | Auto-only correct rate | Mean performance score |
|---|---:|---:|---:|---:|---:|---:|---:|
| code_generation | 50 | 47 | 3 | 0 | 0 | 0.940000 | 0.940000 |
| factual_qa | 50 | 0 | 0 | 50 | 0 |  | 0.861855 |
| math_reasoning | 50 | 40 | 10 | 0 | 0 | 0.800000 | 0.800000 |

## Notes

The factual QA evaluator stores three factual-specific fields:

```text
factual_containment_match
factual_token_f1
performance_score
```

The math evaluator extracts the final answer and checks normalized or symbolic equivalence. If the parser cannot safely compare the answer, the row is labeled incorrect.

The code-generation evaluator loaded HumanEvalPack Python tests and executed extracted code in temporary subprocesses.

## Next Steps

1. After formal perturbed generations exist, run:

```bash
/opt/anaconda3/bin/python src/23_evaluate_rq2_correctness.py \
  --generations outputs/rq1_formal_perturbed_generations.csv \
  --output rq2/outputs/rq2_perturbed_correctness_by_generation.csv \
  --summary rq2/outputs/rq2_perturbed_correctness_summary_by_task.csv
```

2. Combine original correctness, perturbed correctness, and RQ1 semantic-drift results into an item-level RQ2 analysis table.
