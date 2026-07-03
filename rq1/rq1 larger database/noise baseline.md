# RQ1 Larger Database: Noise Baseline

Date analyzed: 2026-07-03

This document records the expanded RQ1 sample-noise baseline analysis using 50 prompts per task type. The purpose of this expanded run is to test whether the LLM output noise baseline is task-dependent with a larger sample than the earlier 10-prompt formal run.

## 1. Data Scale

| Task type | Dataset | Prompts | Outputs per prompt | Total outputs |
|---|---|---:|---:|---:|
| factual_qa | SQuAD V2 | 50 | 5 | 250 |
| math_reasoning | Hendrycks MATH | 50 | 5 | 250 |
| code_generation | HumanEvalPack Python | 50 | 5 | 250 |
| open_ended_writing | Alpaca | 50 | 5 | 250 |

Total original prompts: 200

Total baseline outputs: 1000

## 2. Method

For each prompt, the same original prompt was submitted to the same output-generation LLM five times. Sentence-BERT was then used to compute pairwise similarity among the five outputs for the same prompt.

For each prompt:

```text
number_of_outputs = 5
number_of_pairwise_comparisons = 5 choose 2 = 10
mean_within_prompt_similarity = average pairwise similarity among the 5 outputs
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

The task-level baseline is the average sampling-noise drift across the 50 prompts in each task type.

Similarity model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## 3. Task-Level Baseline Results

| Task type | n prompts | Mean within-prompt similarity | Mean sampling-noise drift | SD sampling-noise drift |
|---|---:|---:|---:|---:|
| factual_qa | 50 | 0.971166 | 0.028834 | 0.037073 |
| code_generation | 50 | 0.938495 | 0.061505 | 0.028832 |
| open_ended_writing | 50 | 0.933042 | 0.066958 | 0.060519 |
| math_reasoning | 50 | 0.907748 | 0.092252 | 0.062151 |

Interpretation:

```text
Lower sampling-noise drift means repeated outputs for the same prompt are more stable.
Higher sampling-noise drift means repeated outputs for the same prompt vary more.
```

In this expanded run, factual QA has the lowest baseline drift, while math reasoning has the highest baseline drift.

## 4. Statistical Tests

| Test | Statistic | p value | Interpretation |
|---|---:|---:|---|
| One-way ANOVA | 13.971748 | 2.7185e-08 | Mean baseline drift differs across task types |
| Kruskal-Wallis | 44.746000 | 1.0477e-09 | Nonparametric check also indicates task-type differences |
| Levene median | 8.953014 | 1.3791e-05 | Variances differ across task types |

Conclusion:

```text
The expanded n=50 result supports the claim that the sample-noise baseline is task-dependent.
```

Because Levene's test is significant, the equal-variance assumption is not fully satisfied. Therefore, the Kruskal-Wallis result should be reported alongside ANOVA as a robustness check.

## 5. Tukey HSD Post-Hoc Comparisons

| Group 1 | Group 2 | Mean difference | p adjusted | 95% CI lower | 95% CI upper | Significant |
|---|---|---:|---:|---:|---:|---|
| code_generation | factual_qa | -0.032700 | 0.0060 | -0.0582 | -0.0071 | Yes |
| code_generation | math_reasoning | 0.030700 | 0.0112 | 0.0052 | 0.0563 | Yes |
| code_generation | open_ended_writing | 0.005500 | 0.9457 | -0.0201 | 0.0310 | No |
| factual_qa | math_reasoning | 0.063400 | < .001 | 0.0379 | 0.0890 | Yes |
| factual_qa | open_ended_writing | 0.038100 | 0.0009 | 0.0126 | 0.0637 | Yes |
| math_reasoning | open_ended_writing | -0.025300 | 0.0536 | -0.0509 | 0.0003 | No |

Summary:

```text
Factual QA is significantly more stable than code generation, math reasoning, and open-ended writing.
Math reasoning is significantly less stable than code generation.
Code generation and open-ended writing are not significantly different from each other.
Math reasoning and open-ended writing are not significantly different from each other at alpha = .05.
```

## 6. Result Files

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
