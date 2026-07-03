# RQ2 Methodology Design

## Research Question

RQ2 asks whether output-level semantic drift is associated with correctness or performance change in tasks with objective correctness criteria.

Operational RQ2:

```text
For factual QA, mathematical reasoning, and code generation, does greater semantic drift after prompt perturbation correspond to a larger drop in task performance? Does this relationship differ across task types?
```

Open-ended writing is excluded because it lacks an objective correctness criterion.

The RQ1 methodology that RQ2 inherits from is maintained separately in:

```text
pioneer methodology.md
```

## Literature Positioning

RQ2 is not a direct replication of a single reference paper. It combines three ideas from the current reference set:

| Method component | Supporting reference direction | Use in RQ2 |
|---|---|---|
| Performance drop under perturbation | PromptRobust / PromptBench; Enhancing LLM Robustness | Use correctness/performance change as the outcome |
| Repeated sampling | Haase et al. | Use repeated generations to estimate performance rates rather than relying on one output |
| Semantic drift | Sentence-BERT; RQ1 pipeline | Test whether drift predicts correctness/performance change |

PDR is used as the literature-aligned performance-drop metric:

```text
PDR = (Performance_original - Performance_perturbed) / Performance_original
```

When `Performance_original = 0`, PDR is undefined, so absolute performance change should also be reported:

```text
absolute_performance_change = Performance_original - Performance_perturbed
```

## Analysis Unit

The recommended RQ2 analysis unit is:

```text
item_id + perturbation_type
```

For each item and perturbation type, compute:

```text
original_performance_rate_or_score
perturbed_performance_rate_or_score
absolute_performance_change
PDR
semantic_drift
task_type
```

This avoids treating repeated outputs from the same item as fully independent observations.

## Shared Correctness/Performance Principle

RQ2 uses task-specific automatic evaluation. The project does not use manual correctness judging because manual review would be too large and difficult to reproduce consistently.

The evaluator should be described as:

```text
automatic, benchmark-style, and conservative
```

This means the evaluator is reproducible, but it may underestimate correctness when an answer is semantically correct without lexical or executable evidence that the automatic rule can detect.

## Task 1: Factual QA

### Dataset

```text
task_type = factual_qa
dataset = SQuAD V2
```

### Evaluation Problem

Factual QA outputs are often full sentences rather than short answer spans. A direct whole-output token F1 score can underestimate correctness because extra explanatory tokens lower precision.

Example:

```text
reference answer = "the City council"
model output = "Since 1990, the President of Warsaw has been elected by the City Council."
```

The answer is present, but whole-output token F1 would be lowered by the additional words in the sentence.

### Recommended Automatic Method

Use a two-step score:

```text
Containment first + token F1 backup
```

Step 1: Normalize the reference answer and model output.

Normalization:

```text
lowercase
remove punctuation
remove English articles: a, an, the
collapse whitespace
```

Step 2: Check normalized containment.

```text
if normalized_reference appears in normalized_output:
    factual_containment_match = true
    factual_score = 1.0
```

Step 3: If containment does not hold, compute SQuAD-style token F1 between the normalized reference answer and the normalized full output.

```text
precision = overlapping_tokens / output_tokens
recall = overlapping_tokens / reference_tokens
token_f1 = 2 * precision * recall / (precision + recall)
```

Then:

```text
if factual_containment_match:
    factual_score = 1.0
else:
    factual_score = token_f1
```

Equivalent compact form:

```text
factual_score = max(containment_score, token_f1)
```

where:

```text
containment_score = 1.0 if normalized_reference appears in normalized_output, else 0.0
```

### Binary Correctness Decision

For factual QA, the current RQ2 design does not assign a binary correctness label. The reason is that factual QA outputs often contain full-sentence answers, partial answer spans, or lexically different but potentially valid formulations. For this task, a continuous automatic performance score is more appropriate than forcing every output into correct/incorrect.

Therefore:

```text
is_correct = blank for factual_qa
performance_score = factual_score
```

This keeps factual QA aligned with SQuAD-style automatic scoring while avoiding an arbitrary threshold.

### Recommended RQ2 Performance Variable

For factual QA:

```text
Performance = mean factual_score across repeated generations
```

Then:

```text
absolute_performance_change = original_mean_factual_score - perturbed_mean_factual_score
PDR = absolute_performance_change / original_mean_factual_score
```

### Limitation

This method is still lexical. It can miss semantically equivalent answers that do not share tokens with the reference answer.

Example:

```text
reference answer = "former captain of the Yale football team"
model output = "Walter Camp"
```

Depending on the original question, the output may be semantically correct, but lexical overlap may be low. Therefore, the factual QA score should be interpreted as a conservative automatic SQuAD-style score, not perfect factual truth.

## Task 2: Mathematical Reasoning

### Dataset

```text
task_type = math_reasoning
dataset = MATH / Hendrycks MATH
```

### Evaluation Problem

Math outputs often include long reasoning traces, but correctness depends primarily on the final answer.

### Recommended Automatic Method

Use final-answer extraction followed by normalized and symbolic comparison.

Extraction priority:

```text
1. LaTeX boxed answer, such as \boxed{...}
2. Phrases such as "final answer is" or "answer is"
3. Last numeric or simple mathematical expression in the output
```

Comparison:

```text
1. Normalize extracted answer and reference answer.
2. If normalized strings match, mark correct.
3. Otherwise, use symbolic/numeric equivalence when possible.
4. If extraction or comparison fails, mark incorrect under the fully automatic conservative policy.
```

### Recommended RQ2 Performance Variable

For math reasoning:

```text
Performance = final-answer correctness rate across repeated generations
```

Each generation receives:

```text
1 = final answer correct
0 = final answer incorrect or unresolved by automatic parser
```

Unlike factual QA, math reasoning keeps a binary correctness label because the task has a specific final answer.

Then:

```text
absolute_performance_change = original_correct_rate - perturbed_correct_rate
PDR = absolute_performance_change / original_correct_rate
```

### Limitation

The parser may fail on structured answers, equivalent expressions written in unusual forms, or answers where the model gives the correct value without a clear final-answer marker. Under the no-manual-review policy, these unresolved cases are labeled incorrect, so the result is conservative.

## Task 3: Code Generation

### Dataset

```text
task_type = code_generation
dataset = HumanEvalPack Python
```

### Evaluation Problem

Code output can be textually different while functionally equivalent. Therefore, textual similarity to the reference solution is not an appropriate correctness metric.

### Recommended Automatic Method

Use functional correctness through unit-test execution.

Procedure:

```text
1. Extract Python code from the model output.
2. Load the corresponding HumanEvalPack test metadata using source_index or source_id.
3. Execute the extracted code with the task's unit tests in a temporary subprocess.
4. Mark correct if all tests pass.
5. Mark incorrect if tests fail, syntax errors occur, runtime errors occur, or execution times out.
```

### Recommended RQ2 Performance Variable

For code generation:

```text
Performance = unit-test pass rate across repeated generations
```

Each generation receives:

```text
1 = passes all tests
0 = fails tests, cannot run, or times out
```

Unlike factual QA, code generation keeps a binary correctness label because functional test execution gives an objective pass/fail outcome.

Then:

```text
absolute_performance_change = original_pass_rate - perturbed_pass_rate
PDR = absolute_performance_change / original_pass_rate
```

### Limitation

Unit tests measure functional behavior only for the tested cases. A solution that passes all provided tests may still fail hidden or untested cases. Conversely, code extraction errors can cause a generated solution to be marked incorrect even if a human could repair minor formatting around the code. This is still the most objective automatic criterion for code generation in the current design.

## Cross-Task RQ2 Outcomes

Primary outcome:

```text
absolute_performance_change
```

Literature-aligned secondary outcome:

```text
PDR
```

Binary robustness outcome, optional:

```text
performance_dropped = perturbed_performance < original_performance
```

Main predictor:

```text
semantic_drift
```

from the RQ1 Sentence-BERT drift analysis.

## Planned Statistical Analysis

Start with descriptive analysis:

```text
mean original performance by task type
mean perturbed performance by task type and perturbation type
mean absolute performance change by task type and perturbation type
mean PDR by task type and perturbation type
```

Then test the RQ2 relationship:

```text
absolute_performance_change ~ semantic_drift
PDR ~ semantic_drift
```

For task-type comparison:

```text
outcome ~ semantic_drift * task_type
```

Because the formal sample is modest, results should be described as exploratory unless the sample is expanded.

## Planned Output Files

Generation-level correctness/performance:

```text
rq2/outputs/rq2_original_correctness_by_generation.csv
rq2/outputs/rq2_perturbed_correctness_by_generation.csv
```

Item-level RQ2 analysis table:

```text
rq2/outputs/rq2_drift_correctness_analysis_by_item.csv
```

Task-level summary:

```text
rq2/outputs/rq2_drift_correctness_summary_by_task.csv
```
