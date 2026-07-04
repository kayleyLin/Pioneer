# RQ2 Final Output

Last rerun: 2026-07-04

Rerun note:

```text
RQ2 outputs were regenerated from the current formal RQ1 files:
outputs/rq1_formal_original_generations.csv
outputs/rq1_formal_perturbed_generations.csv

The perturbed-generation file contains 1000 rows overall, including the 750 objective-task rows used for RQ2.
```

## Research Question

RQ2 asks whether semantic drift is associated with task performance change for tasks with objective evaluation criteria.

Operational version:

```text
For factual QA, mathematical reasoning, and code generation, does greater semantic drift after prompt perturbation correspond to a larger drop in task performance? Does this relationship differ across task types?
```

Open-ended writing is excluded because it lacks an objective correctness criterion.

## Data Used

This RQ2 output uses the currently available formal RQ1 generation files:

```text
outputs/rq1_formal_original_generations.csv
outputs/rq1_formal_perturbed_generations.csv
```

For RQ2, the available perturbed-output set is complete:

```text
3 task types x 10 items x 5 perturbation types x 5 generations = 750 perturbed outputs
```

The original side contains:

```text
3 task types x 10 items x 5 generations = 150 original outputs
```

## Task-Specific Performance Metrics

RQ2 uses task-specific automatic performance metrics rather than forcing all tasks into the same binary correctness label.

| Task type | Performance metric |
|---|---|
| factual_qa | Continuous factual QA score: normalized containment first, SQuAD-style token F1 backup |
| math_reasoning | Final-answer correctness, 1/0 |
| code_generation | HumanEvalPack unit-test pass/fail, 1/0 |

For factual QA, binary correctness is not assigned. The performance score is continuous.

## Main Output Files

Generation-level performance:

```text
rq2/outputs/rq2_original_correctness_by_generation.csv
rq2/outputs/rq2_formal_available_perturbed_performance_by_generation.csv
```

Performance-change tables:

```text
rq2/outputs/rq2_formal_available_performance_change_by_item.csv
rq2/outputs/rq2_formal_available_performance_change_summary.csv
```

Drift-performance tables:

```text
rq2/outputs/rq2_formal_available_drift_performance_by_item.csv
rq2/outputs/rq2_formal_available_drift_performance_summary.csv
```

## Formal Results

### Result 1: Original And Perturbed Performance By Task

How to read this table:

```text
Each row is one task type.
Original performance is the average score under the original prompts.
Perturbed performance is the average score across all five perturbation types.
Absolute performance change = original performance - perturbed performance.
A larger positive value means performance dropped more after perturbation.
```

中文批注：

```text
这张表回答的是：整体来看，每个任务类型在 prompt 被扰动之后表现下降了多少。
factual_qa 的 performance 不是二元 correct/incorrect，而是连续的 factual score。
math_reasoning 和 code_generation 的 performance 是 0/1 correctness 或 pass rate 的平均值。
```

| Task type | Original performance | Perturbed performance | Absolute performance change |
|---|---:|---:|---:|
| code_generation | 0.940000 | 0.852000 | 0.088000 |
| factual_qa | 0.861855 | 0.766904 | 0.094951 |
| math_reasoning | 0.800000 | 0.740000 | 0.060000 |

Interpretation:

```text
All three objective task types show lower average performance after perturbation.
The largest average performance decrease appears in factual QA, followed closely by code generation.
Math reasoning shows a smaller average decrease in this run.
```

### Result 2: Overall Pattern By Perturbation Type

How to read this table:

```text
Each row is one perturbation type, pooled across factual QA, math, and code.
Mean noise-corrected drift measures how much the output meaning changed after subtracting ordinary sampling noise.
Mean absolute performance change measures how much task performance dropped.
If both columns are high, that perturbation changes outputs semantically and also hurts task performance.
```

中文批注：

```text
这张表回答的是：哪一种 prompt perturbation 最危险。
noise-corrected drift 越大，说明输出和原 prompt 输出相比变化越大。
performance change 越大，说明任务表现下降越多。
这里 paraphrasing 同时在两个指标上最高，所以它是当前结果里最强的扰动类型。
```

| Perturbation type | Mean noise-corrected drift | Mean absolute performance change |
|---|---:|---:|
| paraphrasing | 0.050990 | 0.256647 |
| reordering | 0.016125 | 0.059228 |
| formatting_changes | 0.010170 | 0.051278 |
| surface_noise | -0.003515 | 0.019444 |
| context_injection | -0.001012 | 0.018321 |

Interpretation:

```text
Paraphrasing is the strongest perturbation type in both semantic drift and performance decrease.
Reordering and formatting changes show moderate effects.
Context injection and surface noise show small average performance effects in this run.
```

### Result 3: Full Task-By-Perturbation Results

How to read this table:

```text
Each row is one task type plus one perturbation type.
Mean drift is the average noise-corrected semantic drift for that task-perturbation pair.
Mean performance change is the average performance drop for that task-perturbation pair.
Pearson drift-change is the within-cell linear association across the 10 items in that cell.
```

中文批注：

```text
这张表是最细的 summary。
比如 factual_qa + paraphrasing 这一行：
Mean drift = 0.112586，说明 paraphrasing 让 factual QA 输出语义变化很大。
Mean performance change = 0.329941，说明 factual QA 的 performance score 平均下降了约 0.33。
Pearson = 0.480695，说明在这 10 个 factual QA items 里，drift 越大的 item 通常 performance drop 也更大一些。
```

Important caution:

```text
Each Pearson value here is based on only 10 items, so it should be read as descriptive evidence, not strong inferential proof.
```

| Task type | Perturbation | Mean drift | Mean performance change | Pearson drift-change |
|---|---|---:|---:|---:|
| code_generation | paraphrasing | 0.034560 | 0.300000 | 0.650357 |
| code_generation | context_injection | 0.002075 | 0.040000 | -0.206895 |
| code_generation | formatting_changes | 0.004808 | 0.040000 | -0.572964 |
| code_generation | reordering | 0.007690 | 0.020000 | 0.210501 |
| code_generation | surface_noise | 0.001728 | 0.040000 | -0.212427 |
| factual_qa | paraphrasing | 0.112586 | 0.329941 | 0.480695 |
| factual_qa | reordering | 0.041203 | 0.097685 | 0.194632 |
| factual_qa | formatting_changes | 0.026844 | 0.073834 | 0.010064 |
| factual_qa | context_injection | -0.005502 | -0.025036 | 0.061895 |
| factual_qa | surface_noise | -0.006236 | -0.001669 | -0.444451 |
| math_reasoning | paraphrasing | 0.005822 | 0.140000 | -0.015417 |
| math_reasoning | reordering | -0.000518 | 0.060000 | -0.491941 |
| math_reasoning | context_injection | 0.000390 | 0.040000 | -0.546474 |
| math_reasoning | formatting_changes | -0.001141 | 0.040000 | -0.606590 |
| math_reasoning | surface_noise | -0.006036 | 0.020000 | 0.042583 |

### Result 4: Performance Drop Summary Without Drift

This table reports performance change alone, before joining it to semantic drift.

How to read this table:

```text
This table ignores semantic drift and only looks at task performance.
Original performance is the repeated-generation performance under original prompts.
Perturbed performance is the repeated-generation performance under one perturbation type.
Absolute change = original performance - perturbed performance.
Mean PDR is the relative performance drop:
PDR = (original performance - perturbed performance) / original performance.
Items with drop counts how many of the 10 items got worse under that perturbation.
```

中文批注：

```text
这张表回答的是：不管 semantic drift，只看任务表现，哪个 task + perturbation 组合掉分最多。
Absolute change 是绝对下降，比如 0.300000 表示平均表现下降 0.30。
Mean PDR 是相对下降，比如 0.300000 表示相对原始表现下降 30%。
Items with drop 是 10 个 item 里有多少个真的下降。
```

| Task type | Perturbation | Original performance | Perturbed performance | Absolute change | Mean PDR | Items with drop |
|---|---|---:|---:|---:|---:|---:|
| code_generation | context_injection | 0.940000 | 0.900000 | 0.040000 | 0.100000 | 1 |
| code_generation | formatting_changes | 0.940000 | 0.900000 | 0.040000 | 0.100000 | 1 |
| code_generation | paraphrasing | 0.940000 | 0.640000 | 0.300000 | 0.300000 | 3 |
| code_generation | reordering | 0.940000 | 0.920000 | 0.020000 | 0.050000 | 1 |
| code_generation | surface_noise | 0.940000 | 0.900000 | 0.040000 | 0.100000 | 1 |
| factual_qa | context_injection | 0.861855 | 0.886891 | -0.025036 | -0.045574 | 0 |
| factual_qa | formatting_changes | 0.861855 | 0.788021 | 0.073834 | 0.065144 | 1 |
| factual_qa | paraphrasing | 0.861855 | 0.531914 | 0.329941 | 0.419635 | 7 |
| factual_qa | reordering | 0.861855 | 0.764170 | 0.097685 | 0.110271 | 3 |
| factual_qa | surface_noise | 0.861855 | 0.863523 | -0.001669 | -0.006781 | 0 |
| math_reasoning | context_injection | 0.800000 | 0.760000 | 0.040000 | -0.133333 | 1 |
| math_reasoning | formatting_changes | 0.800000 | 0.760000 | 0.040000 | -0.116667 | 2 |
| math_reasoning | paraphrasing | 0.800000 | 0.660000 | 0.140000 | 0.250000 | 3 |
| math_reasoning | reordering | 0.800000 | 0.740000 | 0.060000 | -0.011111 | 3 |
| math_reasoning | surface_noise | 0.800000 | 0.780000 | 0.020000 | 0.027778 | 1 |

Note:

```text
Negative PDR can occur when a perturbed condition performs better than the original condition for some items. This is possible because performance is estimated from repeated stochastic generations.
```

### Result 5: Item-Level Output

The item-level table contains 150 rows:

```text
3 task types x 10 items x 5 perturbation types = 150 item-perturbation rows
```

Each row contains:

```text
item_id
task_type
perturbation_type
original_performance
perturbed_performance
absolute_performance_change
pdr
baseline_similarity
perturbation_similarity
noise_corrected_drift
performance_dropped
```

The item-level table is saved at:

```text
rq2/outputs/rq2_formal_available_drift_performance_by_item.csv
```

中文批注：

```text
前面的表都是 summary。
这个 item-level 文件才是统计分析真正用的数据表。
一行代表一个 item 在一种 perturbation 下的结果。
RQ2 一共有 3 个 task x 10 个 item x 5 个 perturbation type = 150 行。
```

## Answer To RQ2

Based on the currently available formal RQ2 data, semantic drift and performance decrease are not uniformly related across all tasks and perturbation types.

The clearest positive drift-performance pattern appears for paraphrasing:

```text
code_generation paraphrasing: drift = 0.034560, performance change = 0.300000, r = 0.650357
factual_qa paraphrasing: drift = 0.112586, performance change = 0.329941, r = 0.480695
```

This suggests that when paraphrasing produces larger semantic drift, it also tends to correspond to larger performance loss in factual QA and code generation.

However, this pattern is weaker or absent in math reasoning:

```text
math_reasoning paraphrasing: drift = 0.005822, performance change = 0.140000, r = -0.015417
```

Therefore, the association between semantic drift and task performance change appears task-dependent. Semantic drift is most informative for performance loss in factual QA and code generation under paraphrasing, but it is not a consistent predictor across all perturbation types or all tasks.

## Statistical Analysis

A follow-up statistical analysis was run on the 150 item-perturbation rows in:

```text
rq2/outputs/rq2_formal_available_drift_performance_by_item.csv
```

The main outcome was:

```text
absolute_performance_change
```

The main predictor was:

```text
noise_corrected_drift
```

### Descriptive Statistics

How to read this section:

```text
These tables add standard deviation and 95% confidence intervals.
The confidence interval gives a rough uncertainty range for the mean.
If a CI for performance change is mostly above 0, it suggests performance tends to drop.
```

中文批注：

```text
这部分是在 Result 1 和 Result 2 的基础上加统计不确定性。
95% CI 不是最终真理，但可以帮助判断平均值是否稳定。
比如 paraphrasing 的 performance change CI 是 [0.110617, 0.402677]，整个区间都在 0 以上，说明 paraphrasing 的 performance drop 比较稳定。
```

By task type:

| Task type | n | Mean drift | 95% CI drift | Mean performance change | 95% CI performance change | Mean PDR |
|---|---:|---:|---:|---:|---:|---:|
| code_generation | 50 | 0.010172 | [0.002482, 0.017862] | 0.088000 | [0.016140, 0.159860] | 0.130000 |
| factual_qa | 50 | 0.033779 | [0.013030, 0.054528] | 0.094951 | [0.017525, 0.172377] | 0.108539 |
| math_reasoning | 50 | -0.000296 | [-0.006558, 0.005966] | 0.060000 | [-0.005213, 0.125213] | 0.003333 |

By perturbation type:

| Perturbation type | n | Mean drift | 95% CI drift | Mean performance change | 95% CI performance change | Mean PDR |
|---|---:|---:|---:|---:|---:|---:|
| paraphrasing | 30 | 0.050990 | [0.022317, 0.079663] | 0.256647 | [0.110617, 0.402677] | 0.325736 |
| reordering | 30 | 0.016125 | [0.000303, 0.031947] | 0.059228 | [-0.019452, 0.137908] | 0.051818 |
| formatting_changes | 30 | 0.010170 | [-0.005386, 0.025726] | 0.051278 | [-0.027272, 0.129828] | 0.020739 |
| surface_noise | 30 | -0.003515 | [-0.010338, 0.003308] | 0.019444 | [-0.010685, 0.049573] | 0.040765 |
| context_injection | 30 | -0.001012 | [-0.008183, 0.006159] | 0.018321 | [-0.050714, 0.087356] | -0.022612 |

### Correlation Tests

How to read this section:

```text
Correlation tests ask whether items with larger semantic drift also tend to have larger performance drop.
Pearson r tests linear association.
Spearman rho tests rank/monotonic association.
p-value indicates how surprising the observed association would be under no association.
```

中文批注：

```text
这里的两个变量是：
X = noise_corrected_drift
Y = absolute_performance_change

Pearson 显著但 Spearman 不显著，说明可能有线性趋势，但不是非常稳定的单调关系。
按 task 分开看，factual QA 最稳定，因为 Pearson 和 Spearman 都显著。
```

Overall:

| Scope | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| Overall | 150 | 0.327704 | 0.000042 | 0.089610 | 0.275487 |

By task type:

| Task type | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| code_generation | 50 | 0.537373 | 0.000057 | 0.131946 | 0.361031 |
| factual_qa | 50 | 0.481403 | 0.000401 | 0.434788 | 0.001604 |
| math_reasoning | 50 | -0.330193 | 0.019187 | -0.215038 | 0.133694 |

By perturbation type:

| Perturbation type | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| paraphrasing | 30 | 0.436551 | 0.015870 | 0.516404 | 0.003484 |
| reordering | 30 | 0.125978 | 0.507113 | -0.132136 | 0.486399 |
| formatting_changes | 30 | -0.135879 | 0.474025 | -0.337711 | 0.067977 |
| context_injection | 30 | -0.379764 | 0.038458 | -0.253218 | 0.176975 |
| surface_noise | 30 | -0.039084 | 0.837531 | -0.228219 | 0.225138 |

Interpretation:

```text
The overall Pearson correlation is positive and statistically significant, but the overall Spearman correlation is not. This suggests that the relationship is not a simple universal monotonic pattern.

By task type, factual QA shows the most consistent association: both Pearson and Spearman correlations are positive and statistically significant. Code generation shows a significant Pearson correlation but not a significant Spearman correlation. Math reasoning shows a negative Pearson correlation, suggesting a different relationship between embedding-based drift and performance change for math tasks.

By perturbation type, paraphrasing shows the clearest positive association between drift and performance change.
```

### Regression Models

How to read this section:

```text
Regression tests whether semantic drift predicts performance change while optionally controlling for task type and perturbation type.
The coefficient for noise_corrected_drift tells how much performance change is expected to increase when drift increases by 1 unit.
Interaction terms test whether the drift-performance relationship differs by task type.
```

中文批注：

```text
回归模型是为了回答 RQ2 后半句：这个关系在不同 task type 里面是不是一样强。
如果 drift:task_type interaction 显著，说明不同 task 的 drift-performance 关系不一样。
这里 math_reasoning 的 interaction 显著，说明 math 和其他任务的关系明显不同。
```

Three OLS models were fit with heteroscedasticity-robust HC3 standard errors.

Model 1:

```text
absolute_performance_change ~ noise_corrected_drift
```

Key result:

```text
noise_corrected_drift coefficient = 1.6937
p = 0.007
R^2 = 0.107
```

This indicates that higher noise-corrected drift is associated with larger performance decrease in the pooled data.

Model 2:

```text
absolute_performance_change ~ noise_corrected_drift * task_type
```

Key result:

```text
noise_corrected_drift coefficient for code_generation baseline = 5.0216, p = 0.029
math_reasoning interaction = -8.4603, p = 0.002
factual_qa interaction = -3.2252, p = 0.176
R^2 = 0.220
```

This supports the claim that the drift-performance relationship differs by task type, especially for math reasoning.

Model 3:

```text
absolute_performance_change ~ noise_corrected_drift * task_type + perturbation_type
```

Key result:

```text
paraphrasing coefficient = 0.1491, p = 0.047
math_reasoning drift interaction = -7.8521, p = 0.004
noise_corrected_drift main effect = 4.0205, p = 0.097
R^2 = 0.264
```

After controlling for perturbation type, paraphrasing remains associated with larger performance decrease, and math reasoning still shows a significantly different drift-performance relationship.

### Mixed-Effects Robustness Check

A mixed-effects model with item-level random intercepts was also fit:

```text
absolute_performance_change ~ noise_corrected_drift * task_type + perturbation_type + (1 | item_id)
```

The model converged, but the random-effects covariance was singular and the estimated group variance was approximately zero. Therefore, this model should be treated only as a robustness check, not as the main inferential result.

The mixed model still found:

```text
paraphrasing coefficient = 0.139, p = 0.022
noise_corrected_drift coefficient = 5.348, p < 0.001
factual_qa drift interaction = -4.082, p = 0.001
math_reasoning drift interaction = -10.128, p < 0.001
```

Because of the singular random-effects warning, the OLS models with robust standard errors are the primary statistical results.

## Main Takeaways

1. Paraphrasing produced the largest average semantic drift and the largest average performance decrease.
2. Factual QA showed the largest average semantic drift and performance decrease among the three RQ2 tasks.
3. Code generation showed a strong performance decrease under paraphrasing, even though its average drift was lower than factual QA.
4. Math reasoning showed performance decreases, but semantic drift was close to zero on average, suggesting that correctness can change even when embedding-based semantic drift is small.
5. The drift-performance relationship is task-dependent and perturbation-dependent rather than universal.

## Limitations

The Pearson correlations are descriptive because each task-perturbation cell contains only 10 items. They should not be interpreted as strong inferential statistical evidence without a larger sample or a formal regression model.

The factual QA performance score is automatic and lexical. It follows a SQuAD-style logic with a containment-first adaptation for full-sentence LLM outputs, but it may still underestimate semantically correct answers that do not share tokens with the reference answer.

The code-generation metric depends on the available HumanEvalPack tests. Passing tests indicates functional correctness under the provided tests, not exhaustive correctness for all possible inputs.

## Suggested Short Report Wording

```text
For RQ2, I evaluated whether semantic drift is associated with task performance change across factual QA, math reasoning, and code generation. I used task-specific automatic performance metrics: a SQuAD-style continuous factual QA score, final-answer correctness for math, and HumanEvalPack unit-test pass rate for code. Across the current formal RQ2 data, all three tasks showed lower average performance after prompt perturbation. Paraphrasing was the strongest perturbation type, with the largest mean semantic drift and the largest mean performance decrease. The clearest positive drift-performance association appeared for paraphrasing in factual QA and code generation, while math reasoning showed weaker drift despite correctness changes. This suggests that the drift-performance relationship is task-dependent rather than universal.
```
