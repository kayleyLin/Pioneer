# Debug: Does `outputs/` Have The Same Factual QA Paraphrasing Problem?

## Conclusion

Yes. The original GPT/main data under:

```text
outputs/
```

has the same factual QA paraphrasing prompt problem as `qwen/` and `llama/`.

The affected original file is:

```text
outputs/rq1_formal_perturbed_generations_n50_factual_qa.csv
```

For `task_type == factual_qa` and `perturbation_type == paraphrasing`, the 50 item-level prompts are:

| Prompt shape | Item count |
|---|---:|
| question-only, no `Context:` block | 43 |
| full prompt but with `Rewrite the prompt...` prefix | 7 |
| clean full `Context:` + `Question:` prompt | 0 |

So the original `outputs/` factual QA paraphrasing condition is not a clean context-preserving paraphrase condition.

## What Went Wrong

The intended factual QA paraphrasing condition should preserve the original supporting context and only paraphrase the question:

```text
Context: <same original context>

Question: <paraphrased question>
```

But in the original `outputs/` data, most factual QA paraphrasing prompts were sent to GPT as only the paraphrased question:

```text
<paraphrased question>
```

That means the model was no longer doing context-grounded factual QA. It was answering from parametric knowledge, guessing, or using general world knowledge.

## Concrete Example

Item:

```text
factual_qa_10908
```

Reference answer:

```text
the City council
```

Original context-grounded prompt:

```text
Context: The mayor of Warsaw is called President. Generally, in Poland, the mayors of bigger cities are called presidents ...

Question: Who has elected the President of Warsaw since 1990?
```

But the original GPT/main paraphrasing prompt in `outputs/rq1_formal_perturbed_generations_n50_factual_qa.csv` was only:

```text
Since 1990, who has been responsible for electing the President of Warsaw?
```

No context was provided.

GPT old paraphrasing outputs for this item answered:

```text
Since 1990, the President of Warsaw has been elected by the city's residents through direct elections.
```

This differs from the reference/context answer:

```text
the City council
```

When the fixed prompt includes the original context, GPT mostly returns the context-grounded answer:

```text
Since 1990, the President of Warsaw has been elected by the City Council.
```

## Rewrite-prefix Example

Some original factual QA paraphrasing rows did include a context, but they were still contaminated by a rewrite instruction prefix.

Example:

```text
factual_qa_10550
```

Original flawed paraphrasing prompt:

```text
Rewrite the prompt while maintaining its essence:

Context: The First British Empire, established on mercantilism, encompassed territories mainly in North America, the Caribbean, and India...

Question: At what point did Great Britain cede its North American colonies?
```

This is also not a clean QA prompt. The model sees an instruction to rewrite a prompt, not only a factual QA task.

## Effect On `outputs/` Summary

Original GPT/main summary:

```text
outputs/sbert_rq1_n50_perturbation_summary.csv
```

Original result:

| Cell | Rank | Mean noise-corrected drift |
|---|---:|---:|
| `factual_qa + paraphrasing` | 1 | 0.186192 |

After repairing the factual QA paraphrasing prompts and regenerating GPT/main outputs:

```text
outputs/sbert_rq1_n50_perturbation_summary_fixed_factual.csv
```

Fixed result:

| Cell | Rank | Mean noise-corrected drift |
|---|---:|---:|
| `factual_qa + paraphrasing` | 2 | 0.084251 |

The largest fixed GPT/main cell becomes:

```text
open_ended_writing + paraphrasing
```

with mean noise-corrected drift:

```text
0.135086
```

## Files Created For The Fixed `outputs/` Branch

Fixed prompt file:

```text
prompts/rq1_formal_perturbed_prompts_n50_factual_qa_fixed.csv
```

Fixed GPT/main factual QA paraphrasing generations:

```text
outputs/rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv
```

Raw duplicate backup from interrupted OpenAI generation:

```text
outputs/rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed_raw_with_duplicates.csv
```

Merged fixed factual QA perturbed file:

```text
outputs/rq1_formal_perturbed_generations_n50_factual_qa_fixed.csv
```

Fixed SBERT item effects:

```text
outputs/sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv
```

Fixed full item-level SBERT effects:

```text
outputs/sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv
```

Fixed summary:

```text
outputs/sbert_rq1_n50_perturbation_summary_fixed_factual.csv
```

Fixed item analysis table:

```text
outputs/factual_paraphrase_item_table_fixed_factual.csv
```

## Interpretation

The original `outputs/` result should not be interpreted as:

```text
Factual QA is the most sensitive task to clean paraphrasing.
```

It should be interpreted as:

```text
The original factual QA paraphrasing construction often removed or altered the supporting context, and this caused large output drift.
```

After fixing the condition so the context is preserved, factual QA paraphrasing remains affected but is no longer the largest GPT/main drift cell.

## Same Issue In `D:\pioneer_kayley\Pioneer\outputs`

The separate directory:

```text
D:\pioneer_kayley\Pioneer\outputs
```

has the same factual QA paraphrasing problem.

Affected file:

```text
D:\pioneer_kayley\Pioneer\outputs\rq1_formal_perturbed_generations_n50_factual_qa.csv
```

For `task_type == factual_qa` and `perturbation_type == paraphrasing`, the prompt-shape counts are:

| Prompt shape | Item count |
|---|---:|
| question-only, no `Context:` block | 43 |
| full prompt but with `Rewrite the prompt...` prefix | 7 |
| clean full `Context:` + `Question:` prompt | 0 |

Example item:

```text
factual_qa_10908
```

The paraphrasing prompt stored in that directory is only:

```text
Since 1990, who has been responsible for electing the President of Warsaw?
```

This is missing the original supporting context:

```text
Context: The mayor of Warsaw is called President. Generally, in Poland, the mayors of bigger cities are called presidents ...

Question: Who has elected the President of Warsaw since 1990?
```

So for this item, the model was not asked to answer from the SQuAD context. It had to answer from background knowledge or guess from the question alone. That makes this row a context-removal condition, not a clean paraphrasing condition.

A second example from the same directory is:

```text
factual_qa_10550
```

Its paraphrasing prompt begins:

```text
Rewrite the prompt while maintaining its essence:

Context: The First British Empire, established on mercantilism, encompassed territories mainly in North America, the Caribbean, and India...

Question: At what point did Great Britain cede its North American colonies?
```

This one includes a context, but it is contaminated by a rewrite instruction prefix. Therefore it is also not a clean factual QA prompt.
## Math reasoning paraphrasing condition is not fully clean

Checked files:

```text
prompts/rq1_formal_perturbed_prompts_n50_math_reasoning.csv
outputs/math_prompt_perturbation_change_by_item.csv
outputs/math_prompt_perturbation_change_summary.csv
outputs/math_paraphrase_driver_by_item.csv
qwen/outputs/math_paraphrase_driver_by_item.csv
llama/outputs/math_paraphrase_driver_by_item.csv
```

Script used:

```text
src/49_analyze_math_paraphrase_drivers.py
```

### Main finding

The math reasoning `paraphrasing` condition is not a clean pure paraphrase condition.

It is not the same failure mode as the original factual QA issue, where many prompts were sent as question-only prompts without context. For math reasoning, the problem is instead:

```text
paraphrasing = problem wording rewrite + math/numeric cue loss + graph/ASY deletion + Research/Task/Code template artifacts
```

So the math paraphrasing condition mixes normal paraphrasing with task-format/template changes and partial information loss.

### Final counts after parser correction

There are 50 math reasoning paraphrasing prompts.

| Issue type | Count | Rate |
|---|---:|---:|
| `Research Question:` label | 12/50 | 0.24 |
| `Rewrite:` label | 1/50 | 0.02 |
| `Code Signature` / `Task Signature` / fenced code / function signature | 7/50 | 0.14 |
| Any template artifact | 16/50 | 0.32 |
| Original prompt has `[asy]` graph but paraphrase removes `[asy]` graph | 7/50 | 0.14 |
| Template artifact or `[asy]` graph deletion | 18/50 | 0.36 |

These artifacts are specific to the paraphrasing condition:

| Perturbation type | Research Question | Rewrite | Code/signature artifact | Any template artifact | ASY removed |
|---|---:|---:|---:|---:|---:|
| context_injection | 0 | 0 | 0 | 0 | 0 |
| formatting_changes | 0 | 0 | 0 | 0 | 0 |
| paraphrasing | 12 | 1 | 7 | 16 | 7 |
| reordering | 0 | 0 | 0 | 0 | 0 |
| surface_noise | 0 | 0 | 0 | 0 | 0 |

### Prompt-preservation comparison

After correcting the parser so that reordering and formatting wrappers are handled correctly, the prompt-preservation comparison is:

| Perturbation | Prompt content recall | Problem content recall | Math token recall | Number recall | Template artifact rate |
|---|---:|---:|---:|---:|---:|
| paraphrasing | 0.531626 | 0.651399 | 0.770586 | 0.776456 | 0.32 |
| surface_noise | 0.965579 | 0.980866 | 0.995229 | 1.000000 | 0.00 |
| formatting_changes | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.00 |
| context_injection | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.00 |
| reordering | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.00 |

This shows that math paraphrasing is the only perturbation that substantially changes the actual mathematical problem statement and mathematical cue tokens.

### Affected item IDs

Items with template artifact or graph deletion:

```text
math_reasoning_8004
math_reasoning_11297
math_reasoning_4807
math_reasoning_10051
math_reasoning_3402
math_reasoning_1600
math_reasoning_10586
math_reasoning_338
math_reasoning_4531
math_reasoning_4880
math_reasoning_9599
math_reasoning_3336
math_reasoning_4965
math_reasoning_10521
math_reasoning_6164
math_reasoning_1121
math_reasoning_2891
math_reasoning_6746
```

### Examples

#### `math_reasoning_10051`

Original prompt is a geometry problem with an `[asy]` diagram:

```text
Problem: The isosceles triangle and the square shown here have the same area in square units. What is the height of the triangle, $h$, in terms of the side length of the square, $s$? [asy] ... [/asy]

Instruction: Solve the problem and provide the final answer.
```

Paraphrased prompt:

```text
Research Question: Determine the height of an isosceles triangle, $h$, in terms of the side length of a square, $s$, given that they have the same area in square units.

Code Signature:
    ```python
    def calculate_triangle_height(s: float) -> float:
        pass
    ```
```

Problem:

```text
The graph/diagram is removed, and a code-signature prompt is inserted. This is not a clean paraphrase.
```

#### `math_reasoning_4965`

Original prompt:

```text
Problem: What is the area, in square units, of a regular hexagon inscribed in a circle whose area is $324\pi$ square units? Express your answer in simplest radical form.

Instruction: Solve the problem and provide the final answer.
```

Paraphrased prompt:

```text
Calculate the area, in square units, of a regular hexagon that is inscribed in a circle with an area of $324\pi$ square units. Present your solution in simplest radical form.

Code signature:
    ```python
    def calculate_hexagon_area():
    ```
```

Problem:

```text
The math task is contaminated with a coding-style function signature.
```

#### `math_reasoning_11297`

Original prompt includes a full `[asy]` function graph and asks:

```text
What is the value of $g(g(-1))$?
```

Paraphrased prompt:

```text
Determine the value of $g(g(-1))$ based on the provided portion of the function graph. Please provide the final calculated value as the answer.
```

Problem:

```text
The paraphrase refers to a graph but removes the actual `[asy]` graph. The model no longer receives the same information.
```

#### `math_reasoning_3336`

Original prompt includes graphs of four functions in `[asy]`.

Paraphrased prompt:

```text
Rewrite:
Given the graphs of four functions labeled (2) through (5) where the domain of function (3) is $$\{-5,-4,-3,-2,-1,0,1,2\}$$, determine the product of the labels of the invertible functions.
```

Problem:

```text
The prompt contains a `Rewrite:` label and removes the original graph block.
```

#### `math_reasoning_1600`

Original prompt includes a trigonometric graph in `[asy]`.

Paraphrased prompt:

```text
Task: Determine the value of $a$ from the graph of $y = a \sin (bx + c)$ where $a,$ $b,$ and $c$ are positive constants.

Code:
    ```plaintext
    Find the value of $a$ in the equation $y = a \sin (bx + c)$ given the provided graph.
    ```
```

Problem:

```text
The graph is removed, and the paraphrase introduces a code/plaintext block.
```

### Impact on math reasoning paraphrasing result

The current RQ1 math reasoning ranking is:

| Branch | Rank 1 perturbation | Mean NCP | Next perturbation | Gap |
|---|---|---:|---|---:|
| GPT/main | paraphrasing | 0.033969 | context_injection, 0.000989 | 0.032980 |
| Qwen | paraphrasing | 0.067887 | context_injection, 0.014684 | 0.053203 |
| Llama | paraphrasing | 0.057734 | reordering, 0.014406 | 0.043328 |

So paraphrasing is consistently the largest math perturbation across GPT/main, Qwen, and Llama.

But this result should be interpreted cautiously:

```text
math_reasoning + paraphrasing is largest within math because the paraphrase operation is a stronger semantic/surface transformation than the other math perturbations, and because the current math paraphrase prompt set contains non-clean template/framing artifacts and graph-information deletion.
```

### Final conclusion

The math reasoning paraphrasing condition is not clean.

It should be described as:

```text
mixed paraphrase-plus-template-framing sensitivity
```

not as a pure paraphrasing-only effect.

Recommended paper/debug wording:

```text
For math reasoning, paraphrasing is the largest perturbation within the task, but inspection shows that the paraphrasing condition is partially contaminated by prompt-template artifacts and graph-information deletion. Therefore, the current math paraphrasing result should be treated as a mixed paraphrase/framing effect unless the prompts are repaired and regenerated.
```

## Math reasoning paraphrasing data repair completed

Repair scripts:

```text
src/50_fix_math_paraphrase_prompts.py
src/51_merge_fixed_math_paraphrase_generations.py
src/52_recompute_fixed_factual_math_sbert.py
```

Fixed prompt files:

```text
prompts/rq1_formal_perturbed_prompts_n50_math_reasoning_paraphrasing_fixed.csv
prompts/rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv
```

Prompt repair validation:

| Metric | Before | After |
|---|---:|---:|
| math paraphrasing rows | 50 | 50 |
| template artifact rows | 16 | 0 |
| ASY graph removed rows | 7 | 0 |
| ASY graph restored rows | 0 | 7 |

Regenerated fixed paraphrasing outputs:

```text
outputs/rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv
qwen/outputs/rq1_qwen_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv
llama/outputs/rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv
```

Validation:

| Branch | Rows | Unique items | Samples per item | Empty outputs | Template artifact rows | ASY missing rows |
|---|---:|---:|---:|---:|---:|---:|
| GPT/main | 250 | 50 | 5 | 0 | 0 | 0 |
| Qwen | 250 | 50 | 5 | 0 | 0 | 0 |
| Llama | 250 | 50 | 5 | 0 | 0 | 0 |

Merged fixed full math generation files:

```text
outputs/rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv
qwen/outputs/rq1_qwen_perturbed_generations_n50_math_reasoning_fixed.csv
llama/outputs/rq1_llama_perturbed_generations_n50_math_reasoning_fixed.csv
```

Validation:

| Branch | Rows | Unique items | Item-perturbation cells | Bad sample-count cells | Paraphrase artifact rows | Paraphrase ASY missing rows | Empty outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT/main | 1250 | 50 | 250 | 0 | 0 | 0 | 0 |
| Qwen | 1250 | 50 | 250 | 0 | 0 | 0 | 0 |
| Llama | 1250 | 50 | 250 | 0 | 0 | 0 | 0 |

Recomputed SBERT outputs using fixed factual QA and fixed math reasoning:

```text
outputs/sbert_rq1_n50_perturbation_effects_by_item_fixed_factual_math.csv
outputs/sbert_rq1_n50_perturbation_summary_fixed_factual_math.csv
outputs/sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual_math.csv
qwen/outputs/sbert_rq1_n50_perturbation_effects_by_item_fixed_factual_math.csv
qwen/outputs/sbert_rq1_n50_perturbation_summary_fixed_factual_math.csv
qwen/outputs/sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual_math.csv
llama/outputs/sbert_rq1_n50_perturbation_effects_by_item_fixed_factual_math.csv
llama/outputs/sbert_rq1_n50_perturbation_summary_fixed_factual_math.csv
llama/outputs/sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual_math.csv
```

SBERT validation:

| Branch | by_item rows | summary rows |
|---|---:|---:|
| GPT/main | 1000 | 20 |
| Qwen | 1000 | 20 |
| Llama | 1000 | 20 |

Math reasoning paraphrasing NCP before vs after repair:

| Branch | Before fixed math | After fixed math | Delta |
|---|---:|---:|---:|
| GPT/main | 0.033969 | 0.015392 | -0.018577 |
| Qwen | 0.067887 | 0.040872 | -0.027015 |
| Llama | 0.057734 | 0.041361 | -0.016373 |

Fixed math reasoning ranking:

| Branch | Rank 1 perturbation | Mean NCP | Next perturbation | Gap |
|---|---|---:|---|---:|
| GPT/main | paraphrasing | 0.015392 | context_injection, 0.000989 | 0.014403 |
| Qwen | paraphrasing | 0.040872 | context_injection, 0.014684 | 0.026188 |
| Llama | paraphrasing | 0.041361 | reordering, 0.014406 | 0.026955 |

Conclusion after repair:

```text
All three model branches had the same math paraphrasing prompt-quality issue because they used the same math paraphrase prompt source. The prompt data and regenerated outputs have now been repaired for GPT/main, Qwen, and Llama. After repair, math_reasoning + paraphrasing remains the largest math perturbation in all three branches, but its magnitude is lower than before.
```
