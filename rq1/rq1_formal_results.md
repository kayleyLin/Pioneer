# RQ1 Formal Results Record

## 1. Purpose

This document records the formal RQ1 results and the constants used during data generation. It is intended as the main result log for the current formal RQ1 run.

RQ1:

```text
After applying a sampling-noise correction, is the ranking of the five perturbation types by their effect on output semantic similarity consistent across factual QA, math reasoning, code generation, and open-ended writing?
```

## 2. Fixed Data-Generation Constants

### 2.1 Output-Generation Model

| Setting | Value |
|---|---|
| provider | OpenAI API |
| output-generation model | gpt-4o-mini |
| active research question | RQ1 only |
| system prompt | You are a helpful assistant. Answer the user's prompt directly. |
| temperature | 0.7 |
| top_p | 0.9 |
| repeated generations per prompt | 5 |
| max_output_tokens | not set |

The same output-generation model and generation settings were used for original prompts and perturbed prompts.

### 2.2 Paraphrase-Generation Model

| Setting | Value |
|---|---|
| paraphrase-generation provider | OpenAI API |
| paraphrase-generation model | gpt-3.5-turbo |
| method source | Reference 3 POSIX |

Only the paraphrasing perturbation used an LLM for perturbation construction. The other perturbation types used rule-based, template-based, or fixed-bank procedures.

### 2.3 Sampling Constants

| Setting | Value |
|---|---|
| sampling method | stratified random sampling by task type |
| random seed | 20260623 |
| prompts per task type | 10 |
| task types | 4 |
| total original prompts | 40 |

### 2.4 Dataset Sources

| Task type | Dataset | Number of original prompts |
|---|---|---:|
| factual_qa | rajpurkar/squad_v2 | 10 |
| math_reasoning | nlile/hendrycks-MATH-benchmark | 10 |
| code_generation | bigcode/humanevalpack | 10 |
| open_ended_writing | tatsu-lab/alpaca | 10 |

## 3. Formal Data Scale

| File | Rows | Meaning |
|---|---:|---|
| prompts/rq1_sampled_original_prompts.csv | 40 | Formal original prompts |
| prompts/rq1_formal_perturbed_prompts.csv | 200 | 40 prompts x 5 perturbation types |
| outputs/rq1_formal_original_generations.csv | 200 | 40 original prompts x 5 generations |
| outputs/rq1_formal_perturbed_generations.csv | 1000 | 200 perturbed prompts x 5 generations |

Perturbation counts:

| Perturbation type | Number of perturbed prompts | Number of generated outputs |
|---|---:|---:|
| paraphrasing | 40 | 200 |
| reordering | 40 | 200 |
| formatting_changes | 40 | 200 |
| context_injection | 40 | 200 |
| surface_noise | 40 | 200 |

## 4. Similarity Metric

All formal RQ1 similarity results use Sentence-BERT cosine similarity:

```text
sentence-transformers/all-MiniLM-L6-v2
```

For each original prompt:

```text
number_of_original_outputs = 5
number_of_within-prompt_pairs = 5 choose 2 = 10
mean_within_prompt_similarity = average pairwise similarity among repeated original outputs
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

For each perturbation:

```text
perturbation_similarity = average cross-similarity between original outputs and perturbed outputs
uncorrected_drift = 1 - perturbation_similarity
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

## 5. Formal Sampling-Noise Baseline

| Task type | n items | Mean within-prompt similarity | Mean sampling-noise drift | SD sampling-noise drift |
|---|---:|---:|---:|---:|
| code_generation | 10 | 0.931653 | 0.068347 | 0.028244 |
| factual_qa | 10 | 0.951691 | 0.048309 | 0.064513 |
| math_reasoning | 10 | 0.913485 | 0.086515 | 0.044816 |
| open_ended_writing | 10 | 0.921510 | 0.078490 | 0.072555 |

Baseline interpretation:

```text
math_reasoning had the highest natural sampling-noise drift.
factual_qa had the lowest natural sampling-noise drift.
This confirms that task type matters even before perturbations are introduced.
```

## 6. Perturbation Drift Before Baseline Correction

Uncorrected drift:

```text
uncorrected_drift = 1 - perturbation_similarity
```

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.102908 | 0.160896 | 0.092337 | 0.253682 |
| reordering | 0.076037 | 0.089512 | 0.085997 | 0.114277 |
| formatting_changes | 0.073155 | 0.075153 | 0.085373 | 0.081465 |
| context_injection | 0.070422 | 0.042807 | 0.086905 | 0.097173 |
| surface_noise | 0.070075 | 0.042073 | 0.080479 | 0.080503 |

Before correction, many perturbation-task combinations show non-trivial drift. However, these values include both perturbation-induced change and ordinary same-prompt sampling noise.

## 7. Perturbation Drift After Baseline Correction

Noise-corrected drift:

```text
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.034560 | 0.112586 | 0.005822 | 0.175192 |
| reordering | 0.007690 | 0.041203 | -0.000518 | 0.035788 |
| formatting_changes | 0.004808 | 0.026844 | -0.001141 | 0.002976 |
| context_injection | 0.002075 | -0.005502 | 0.000390 | 0.018683 |
| surface_noise | 0.001728 | -0.006236 | -0.006036 | 0.002013 |

After correction, many effects shrink toward zero. This shows that some of the raw perturbation drift was ordinary sampling noise rather than a perturbation-specific effect.

## 8. Main Formal Findings

1. `paraphrasing` produced the largest positive noise-corrected drift for `open_ended_writing`, `factual_qa`, and `code_generation`.
2. `math_reasoning` showed near-zero noise-corrected drift across all five perturbation types.
3. `surface_noise` and `context_injection` generally produced small or negative drift values after correction.
4. The perturbation ranking was not uniform across task types, supporting the interpretation that prompt sensitivity is task-dependent.
5. Baseline correction materially changed the apparent effect size, especially for perturbations that looked moderately large before correction but became near-zero after correction.

Strongest corrected effects:

| Rank | Task type | Perturbation type | Noise-corrected drift |
|---:|---|---|---:|
| 1 | open_ended_writing | paraphrasing | 0.175192 |
| 2 | factual_qa | paraphrasing | 0.112586 |
| 3 | factual_qa | reordering | 0.041203 |
| 4 | open_ended_writing | reordering | 0.035788 |
| 5 | code_generation | paraphrasing | 0.034560 |

## 9. Whether The Results Match Expectations

Overall, the formal RQ1 results mostly match the methodological expectations, with one important nuance.

### 9.1 Expected Pattern 1: Baseline Correction Should Reduce Apparent Perturbation Effects

This expectation is strongly supported.

Before baseline correction, most perturbation-task combinations showed moderate raw drift. For example, raw drift for `reordering` ranged from 0.076037 to 0.114277 across task types. After baseline correction, the same values dropped to a much smaller range, from -0.000518 to 0.041203.

This matches the central logic of the project:

```text
Some apparent perturbation effects are actually ordinary sampling noise.
Baseline correction should therefore shrink many raw drift estimates.
```

The before/after heatmaps support this expectation clearly. The corrected heatmap is more selective: only some perturbation-task combinations remain meaningfully positive.

### 9.2 Expected Pattern 2: Open-Ended Writing Should Be More Sensitive Than Constrained Tasks

This expectation is partly supported.

`open_ended_writing` had the strongest corrected effect overall:

```text
open_ended_writing + paraphrasing = 0.175192
```

This fits the expectation that open-ended tasks have a larger valid response space. If the prompt is paraphrased, the model may choose a different angle, tone, structure, or content emphasis while still answering appropriately. Therefore, open-ended writing is expected to show larger semantic drift than more constrained tasks.

However, open-ended writing did not have the highest baseline sampling-noise drift. The highest baseline drift was in `math_reasoning`:

```text
math_reasoning baseline drift = 0.086515
open_ended_writing baseline drift = 0.078490
```

This means the expected relationship is not simply "open-ended tasks are always the noisiest." In this sample, math reasoning outputs varied substantially across repeated generations, likely because the model could produce different reasoning paths, explanation lengths, or solution formats even when the final answer target was fixed.

### 9.3 Expected Pattern 3: Paraphrasing Should Be One Of The Stronger Natural Perturbations

This expectation is strongly supported.

After baseline correction, `paraphrasing` was the largest perturbation for three of the four task types:

| Task type | Largest corrected perturbation | Value |
|---|---|---:|
| code_generation | paraphrasing | 0.034560 |
| factual_qa | paraphrasing | 0.112586 |
| open_ended_writing | paraphrasing | 0.175192 |

This is reasonable because paraphrasing changes the wording of the entire prompt while preserving meaning. Even if the task is semantically equivalent, the changed wording may activate different response patterns in the model. This result is consistent with the broader prompt-sensitivity literature, especially POSIX-style prompt paraphrasing.

### 9.4 Expected Pattern 4: Surface Noise Should Have Smaller Effects If It Is Mild And Meaning-Preserving

This expectation is supported.

After correction, `surface_noise` produced very small or negative drift:

| Task type | Surface-noise corrected drift |
|---|---:|
| code_generation | 0.001728 |
| factual_qa | -0.006236 |
| math_reasoning | -0.006036 |
| open_ended_writing | 0.002013 |

This matches the design choice to use mild, realistic surface noise rather than adversarial corruption. Because numbers, formulas, entities, code signatures, and answer-critical text were protected, the model could usually recover the intended task.

### 9.5 Expected Pattern 5: Perturbation Effects Should Be Task-Dependent

This expectation is supported.

The corrected heatmap shows different perturbation rankings by task. For example:

```text
factual_qa: paraphrasing > reordering > formatting_changes
math_reasoning: all perturbations are close to zero
open_ended_writing: paraphrasing is much larger than the other perturbations
code_generation: paraphrasing is largest, but the overall magnitude is modest
```

This supports the RQ1 hypothesis that perturbation sensitivity is not uniform across task types.

### 9.6 Main Unexpected Or Nuanced Result

The main nuance is that `math_reasoning` had the highest baseline sampling-noise drift but almost no corrected perturbation effect.

This suggests that math reasoning outputs may vary naturally across repeated generations, but the five prompt perturbations used here did not push the outputs much farther away than that ordinary variation. In other words:

```text
math_reasoning is noisy, but not especially perturbation-sensitive under these natural perturbations.
```

This is an important distinction. It shows why separating baseline sampling noise from perturbation effect is methodologically useful.

## 10. Statistical Significance Checks

Because several standard deviations are large relative to the mean drift values, the heatmap should be treated as a descriptive result unless supported by statistical testing. I therefore added statistical checks to examine whether the observed differences are stable enough to interpret.

### 10.1 Baseline Differences Across Task Types

The sampling-noise baseline has independent task groups, so one-way ANOVA, Kruskal-Wallis, and Tukey HSD were used as checks.

| Test | Statistic | p value | Interpretation |
|---|---:|---:|---|
| One-way ANOVA | 0.890911 | 0.455153 | No significant mean baseline difference across task types |
| Kruskal-Wallis | 3.501374 | 0.320584 | Nonparametric check also not significant |
| Levene median test | 2.090607 | 0.118625 | No significant variance inequality detected |

Tukey HSD found no significant pairwise task differences in baseline drift at alpha = 0.05.

Interpretation:

```text
Although the baseline means differ numerically, the current sample does not provide strong statistical evidence that task-level baseline drift differs across task types.
```

### 10.2 Perturbation Differences Within Each Task Type

For perturbation effects, Tukey HSD is not the best primary test because each task uses the same 10 prompts across all five perturbation types. This is a repeated-measures structure. Therefore, the more appropriate test is:

```text
Friedman repeated-measures test by task type
paired Wilcoxon signed-rank post-hoc tests with Holm correction
```

Friedman test results:

| Task type | n items | Friedman statistic | p value | Interpretation |
|---|---:|---:|---:|---|
| code_generation | 10 | 9.040000 | 0.060107 | Not statistically significant at 0.05 |
| factual_qa | 10 | 21.240642 | 0.000284 | Significant perturbation-type differences |
| math_reasoning | 10 | 5.120000 | 0.275205 | No significant perturbation-type differences |
| open_ended_writing | 10 | 12.840909 | 0.012080 | Significant overall perturbation-type differences |

Significant Holm-corrected pairwise tests:

| Task type | Pairwise comparison | p value | Holm-adjusted p value |
|---|---|---:|---:|
| factual_qa | paraphrasing vs context_injection | 0.001953 | 0.019531 |
| factual_qa | paraphrasing vs surface_noise | 0.003906 | 0.035156 |
| factual_qa | reordering vs surface_noise | 0.005859 | 0.046875 |

Interpretation:

```text
The strongest inferential support is for factual QA, where paraphrasing and reordering differ significantly from weaker perturbations after multiple-comparison correction.
Open-ended writing has a significant overall Friedman test, but no individual pairwise comparison survives Holm correction in the current n=10 sample.
Math reasoning does not show significant perturbation-type differences, which matches the near-zero corrected heatmap values.
Code generation does not reach statistical significance at the 0.05 threshold. Any apparent differences among perturbation types for code generation should therefore be treated as descriptive only in the current sample.
```

### 10.3 Should Tukey Be Used?

Tukey HSD is appropriate for independent group comparisons, such as checking whether baseline drift differs across task types. It is less appropriate for comparing perturbation types within the same task because the same prompts are reused across perturbation conditions.

Recommended reporting:

```text
Use Tukey HSD only as a baseline task-level post-hoc check.
Use Friedman + paired Wilcoxon with Holm correction for perturbation-type comparisons within each task.
```

Statistical output files:

```text
outputs/rq1_formal_significance_baseline_tests.csv
outputs/rq1_formal_significance_baseline_tukey.csv
outputs/rq1_formal_significance_perturbation_friedman.csv
outputs/rq1_formal_significance_perturbation_pairwise_wilcoxon.csv
```

Statistical script:

```text
src/24_analyze_rq1_formal_significance.py
```

## 11. Deeper Robustness Checks

Because the heatmap does not show a perfectly clean monotonic trend, additional robustness checks were added:

```text
1. Bootstrap 95% confidence intervals for each task x perturbation cell.
2. One-sample tests asking whether corrected drift is greater than zero.
3. Bootstrap rank-stability analysis asking how often each perturbation ranks first within each task.
```

### 11.1 Cells With Bootstrap Confidence Intervals Above Zero

The following cells had bootstrap confidence intervals that excluded zero:

| Task type | Perturbation type | Mean corrected drift | Bootstrap 95% CI | Holm-corrected Wilcoxon significant? |
|---|---|---:|---|---|
| code_generation | paraphrasing | 0.034560 | [0.011376, 0.064679] | No |
| factual_qa | paraphrasing | 0.112586 | [0.058831, 0.176155] | Yes |
| factual_qa | reordering | 0.041203 | [0.011378, 0.081643] | No |
| open_ended_writing | paraphrasing | 0.175192 | [0.055894, 0.308182] | No |

Interpretation:

```text
The most robust single cell is factual_qa + paraphrasing, because it both has a bootstrap CI above zero and survives Holm-corrected one-sample Wilcoxon testing.
Several other cells have positive bootstrap intervals but do not survive Holm correction, so they should be described as suggestive rather than conclusive.
```

### 11.2 Bootstrap Rank Stability

Rank stability estimates how often each perturbation becomes the largest mean corrected drift after resampling prompts within a task type.

| Task type | Most stable top-ranked perturbation | Top-rank probability |
|---|---|---:|
| code_generation | paraphrasing | 0.9890 |
| factual_qa | paraphrasing | 0.9618 |
| math_reasoning | paraphrasing | 0.7595 |
| open_ended_writing | paraphrasing | 0.9595 |

Interpretation:

```text
Even though the absolute effect size for math_reasoning is very small, paraphrasing is still often the largest perturbation within bootstrap resamples.
For code_generation, factual_qa, and open_ended_writing, paraphrasing is both the largest observed perturbation and highly rank-stable.
```

### 11.3 Updated Interpretation After Deeper Statistics

The deeper statistics make the result more nuanced:

```text
The cleanest result is not that every perturbation differs significantly from every other perturbation.
The cleaner result is that paraphrasing is the most rank-stable perturbation, while many smaller perturbation effects shrink toward zero after baseline correction.
```

Therefore, the safest formal claim is:

```text
Baseline correction substantially reduces apparent perturbation effects.
Paraphrasing is the most consistently high-ranking perturbation across task types.
However, strong inferential support is clearest for factual QA, while other task-level patterns should be interpreted cautiously because of small n and overlapping variability.
```

Deeper statistics output files:

```text
outputs/rq1_formal_deeper_cell_statistics.csv
outputs/rq1_formal_deeper_rank_stability.csv
```

Deeper statistics script:

```text
src/25_analyze_rq1_formal_deeper_statistics.py
```

## 12. Generated Figures

The figure-generation code is stored in:

```text
rq1_visualization_code.md
```

Generated figures:

```text
figures/rq1_uncorrected_drift_heatmap.png
figures/rq1_noise_corrected_drift_heatmap.png
figures/rq1_uncorrected_vs_corrected_heatmaps.png
```

Generated heatmap CSV files:

```text
outputs/sbert_rq1_formal_uncorrected_heatmap_drift.csv
outputs/sbert_rq1_formal_corrected_heatmap_drift.csv
outputs/sbert_rq1_formal_heatmap_noise_corrected_drift.csv
```

## 13. Formal Result Files

Baseline files:

```text
outputs/sbert_rq1_formal_baseline_by_item.csv
outputs/sbert_rq1_formal_baseline_by_task.csv
```

Perturbation effect files:

```text
outputs/sbert_rq1_formal_perturbation_effects_by_item.csv
outputs/sbert_rq1_formal_perturbation_summary.csv
outputs/sbert_rq1_formal_heatmap_noise_corrected_drift.csv
outputs/sbert_rq1_formal_uncorrected_perturbation_summary.csv
outputs/sbert_rq1_formal_uncorrected_heatmap_drift.csv
outputs/sbert_rq1_formal_corrected_heatmap_drift.csv
```

Generation files:

```text
outputs/rq1_formal_original_generations.csv
outputs/rq1_formal_perturbed_generations.csv
```

Prompt files:

```text
prompts/rq1_sampled_original_prompts.csv
prompts/rq1_formal_perturbed_prompts.csv
```

Scripts:

```text
src/06_sample_benchmark_prompts.py
src/07_generate_rq1_outputs_openai.py
src/11_generate_rq1b_perturbed_outputs_openai.py
src/19_create_rq1_perturbed_prompts.py
src/22_analyze_rq1_formal_baseline_sbert.py
src/23_analyze_rq1_formal_perturbations_sbert.py
src/24_analyze_rq1_formal_significance.py
src/25_analyze_rq1_formal_deeper_statistics.py
```

## 14. Current Limitations

```text
Only one output-generation model was used.
The formal sample uses 10 prompts per task type, which is larger than the pilot but still modest.
The current formal RQ1 result uses Sentence-BERT semantic similarity only.
Correctness-based evaluation such as PDR has not yet been added to the formal results.
Open-ended writing does not have a reference answer, so correctness/PDR is not appropriate for that task.
```

## 15. Expanded N=50 Sample-Noise Baseline

Date analyzed: 2026-07-03

Purpose:

```text
The first formal baseline run used 10 prompts per task type. Because the task-level
standard deviations were relatively large and the ANOVA did not show a significant
task-type effect, the baseline was expanded to 50 prompts per task type.
```

Expanded baseline data:

| Task type | Dataset | Prompts | Outputs per prompt | Total outputs |
|---|---|---:|---:|---:|
| factual_qa | SQuAD V2 | 50 | 5 | 250 |
| math_reasoning | Hendrycks MATH | 50 | 5 | 250 |
| code_generation | HumanEvalPack Python | 50 | 5 | 250 |
| open_ended_writing | Alpaca | 50 | 5 | 250 |

Task-level Sentence-BERT baseline:

| Task type | n prompts | Mean within-prompt similarity | Mean sampling-noise drift | SD sampling-noise drift |
|---|---:|---:|---:|---:|
| factual_qa | 50 | 0.971166 | 0.028834 | 0.037073 |
| code_generation | 50 | 0.938495 | 0.061505 | 0.028832 |
| open_ended_writing | 50 | 0.933042 | 0.066958 | 0.060519 |
| math_reasoning | 50 | 0.907748 | 0.092252 | 0.062151 |

Statistical tests:

| Test | Statistic | p value | Interpretation |
|---|---:|---:|---|
| One-way ANOVA | 13.971748 | 2.7185e-08 | Mean baseline drift differs across task types |
| Kruskal-Wallis | 44.746000 | 1.0477e-09 | Nonparametric check also indicates task-type differences |
| Levene median | 8.953014 | 1.3791e-05 | Variances differ across task types |

Tukey HSD post-hoc comparisons:

| Group 1 | Group 2 | Mean difference | p adjusted | 95% CI lower | 95% CI upper | Significant |
|---|---|---:|---:|---:|---:|---|
| code_generation | factual_qa | -0.032700 | 0.0060 | -0.0582 | -0.0071 | Yes |
| code_generation | math_reasoning | 0.030700 | 0.0112 | 0.0052 | 0.0563 | Yes |
| code_generation | open_ended_writing | 0.005500 | 0.9457 | -0.0201 | 0.0310 | No |
| factual_qa | math_reasoning | 0.063400 | < .001 | 0.0379 | 0.0890 | Yes |
| factual_qa | open_ended_writing | 0.038100 | 0.0009 | 0.0126 | 0.0637 | Yes |
| math_reasoning | open_ended_writing | -0.025300 | 0.0536 | -0.0509 | 0.0003 | No |

Interpretation:

```text
The expanded n=50 baseline provides stronger evidence that the LLM sampling-noise
baseline is task-dependent. Factual QA has the lowest average sampling-noise drift,
indicating the most stable repeated outputs. Math reasoning has the highest average
sampling-noise drift, indicating the least stable repeated outputs in this run.
Code generation and open-ended writing fall between these two extremes and are not
significantly different from each other in the Tukey comparison.

Because Levene's test is significant, the equal-variance assumption is not fully
satisfied. Therefore, the Kruskal-Wallis result should be reported alongside ANOVA
as a robustness check. The substantive conclusion remains the same: baseline
variation differs by task type.
```

Expanded baseline files:

```text
outputs/rq1_formal_original_generations_n50_factual_qa.csv
outputs/rq1_formal_original_generations_n50_math_reasoning.csv
outputs/rq1_formal_original_generations_n50_code_generation.csv
outputs/rq1_formal_original_generations_n50_open_ended_writing.csv
outputs/sbert_rq1_n50_baseline_by_item.csv
outputs/sbert_rq1_n50_baseline_by_task.csv
outputs/rq1_n50_baseline_significance_tests.csv
outputs/rq1_n50_baseline_tukey.csv
src/31_analyze_rq1_n50_baseline_sbert.py
```
