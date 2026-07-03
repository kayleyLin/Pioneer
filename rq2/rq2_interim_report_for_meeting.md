# RQ2 Interim Report For Meeting

## Current Status

RQ2 is being prepared as an extension of the RQ1 pipeline. The final RQ2 analysis will require the completed RQ1 perturbed-output results and semantic-drift table, but several parts can already be reported:

```text
1. RQ2 task scope has been fixed.
2. Task-specific performance metrics have been defined.
3. The automatic evaluator has been implemented.
4. Original-prompt outputs have already been evaluated.
```

## RQ2 Question

RQ2 asks whether semantic drift is associated with task performance change in tasks with objective evaluation criteria.

Operational version:

```text
For factual QA, mathematical reasoning, and code generation, does greater semantic drift after prompt perturbation correspond to a larger drop in task performance? Does this relationship differ across task types?
```

Open-ended writing is excluded because it does not have an objective correctness criterion.

## Task-Specific Performance Metrics

RQ2 does not force all tasks into the same binary correctness metric. Instead, each task uses the most appropriate automatic performance measure.

| Task type | Dataset | Performance metric |
|---|---|---|
| factual_qa | SQuAD V2 | Continuous factual QA score using containment-first token F1 |
| math_reasoning | MATH / Hendrycks MATH | Final-answer correctness, 1/0 |
| code_generation | HumanEvalPack Python | Unit-test pass/fail, 1/0 |

## Factual QA Method

Factual QA uses a SQuAD-style automatic lexical score, adapted for full-sentence LLM outputs.

Procedure:

```text
1. Normalize the reference answer and model output.
2. If the normalized reference answer appears in the normalized output, assign performance_score = 1.0.
3. Otherwise, compute SQuAD-style token F1 between the reference answer and the full model output.
4. Do not assign binary correctness for factual QA.
```

This avoids forcing long generated answers into a strict correct/incorrect label while still keeping evaluation automatic and reproducible.

Limitation:

```text
The score is lexical, not a full semantic answer-equivalence judgment. It may underestimate semantically correct answers that do not share tokens with the reference.
```

## Math Reasoning Method

Math reasoning uses final-answer evaluation.

Procedure:

```text
1. Extract the final answer from boxed answers, final-answer phrases, or the last numeric expression.
2. Normalize the extracted answer and reference answer.
3. Use symbolic or numeric equivalence when possible.
4. Assign performance_score = 1 for correct and 0 for incorrect.
```

## Code Generation Method

Code generation uses functional correctness.

Procedure:

```text
1. Extract Python code from the generated output.
2. Load the corresponding HumanEvalPack tests.
3. Run the code in a temporary subprocess.
4. Assign performance_score = 1 if all tests pass, otherwise 0.
```

## Current Original-Prompt Baseline

The current evaluator has already been run on:

```text
outputs/rq1_formal_original_generations.csv
```

This includes 150 outputs for RQ2:

```text
3 task types x 10 prompts x 5 repeated generations
```

Current original-output performance:

| Task type | n outputs | Binary correct rate | Mean performance score |
|---|---:|---:|---:|
| factual_qa | 50 | Not assigned | 0.861855 |
| math_reasoning | 50 | 0.800000 | 0.800000 |
| code_generation | 50 | 0.940000 | 0.940000 |

Interpretation:

```text
The original-prompt baseline shows that the model performs relatively strongly on code generation and factual QA under the current automatic metrics, while math reasoning has a lower original-prompt correctness rate.
```

## Planned Performance-Drop Analysis

After perturbed outputs and semantic drift values are available, RQ2 will compute performance change for each item and perturbation type.

Primary outcome:

```text
absolute_performance_change = original_performance - perturbed_performance
```

Literature-aligned secondary outcome:

```text
PDR = (original_performance - perturbed_performance) / original_performance
```

If `original_performance = 0`, PDR is undefined, so absolute performance change will still be reported.

## What Cannot Be Claimed Yet

The final RQ2 relationship cannot be answered yet because the following files are still needed:

```text
outputs/rq1_formal_perturbed_generations.csv
outputs/sbert_rq1_formal_perturbation_effects_by_item.csv
```

Therefore, tonight's report should not claim whether semantic drift predicts correctness or performance drop. The correct claim is:

```text
RQ2 methodology and original-output baseline are ready; final drift-performance association analysis will follow after RQ1 perturbed-generation and SBERT drift outputs are complete.
```

## Suggested Meeting Summary

```text
For RQ2, I have separated correctness/performance evaluation by task type rather than forcing all tasks into a single binary label. Factual QA uses a SQuAD-style continuous score adapted for full-sentence LLM outputs, math uses final-answer correctness, and code uses HumanEvalPack unit-test pass rate. The evaluator is fully automatic and has already been applied to the original-prompt generations. The original baseline is 0.861855 for factual QA, 0.800000 for math, and 0.940000 for code. The final RQ2 analysis will connect these performance measures to semantic drift and perturbation type once the RQ1 perturbed-output and SBERT drift results are available.
```
