# Input Similarity vs Output Similarity

## 1. Purpose

This document records the secondary RQ1 analysis on the relationship between input similarity and output similarity.

Main question:

```text
If a perturbed prompt remains more similar to the original prompt, does the model also produce outputs that remain more similar to the original outputs?
```

This analysis was added because the main perturbation heatmap does not show a simple trend across all task types and perturbation types. Input-output similarity may help explain part of that variation.

## 2. Definitions

Input similarity:

```text
Sentence-BERT cosine similarity between original_prompt and perturbed_prompt
```

Output similarity:

```text
Mean cross-similarity between original-prompt outputs and perturbed-prompt outputs
```

Similarity model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Analysis unit:

```text
item-level perturbation row
```

This means one original prompt under one perturbation type. In the formal RQ1 dataset:

```text
40 original prompts x 5 perturbation types = 200 item-level perturbation rows
```

Each item-level perturbation row summarizes:

```text
5 original outputs
5 perturbed outputs
25 original-vs-perturbed output comparisons
```

## 3. What Pearson And Spearman Mean

| Metric | What it measures | Question it answers |
|---|---|---|
| Pearson r | Linear correlation strength | As input similarity increases, does output similarity increase in an approximately straight-line pattern? |
| Pearson p | Statistical significance of Pearson r | Is the linear relationship unlikely to be random noise? |
| Spearman rho | Rank-based / monotonic correlation strength | Do higher-ranked input similarities generally correspond to higher-ranked output similarities? |
| Spearman p | Statistical significance of Spearman rho | Is the rank-based relationship unlikely to be random noise? |

Short version:

```text
Pearson = linear relationship
Spearman = monotonic / rank relationship
p value = whether the observed relationship is statistically significant
```

Approximate interpretation of r / rho:

| r or rho value | Interpretation |
|---:|---|
| 0 | no correlation |
| 0.1 to 0.3 | weak correlation |
| 0.3 to 0.5 | moderate correlation |
| 0.5+ | stronger correlation |
| negative value | one variable increases while the other tends to decrease |

Important distinction:

```text
A statistically significant p value does not mean the relationship is strong.
It means the observed relationship is unlikely to be random under the null hypothesis.
```

## 4. Pilot Analysis

The pilot used a stratified sample:

```text
2 rows per task_type x perturbation_type cell
4 task types x 5 perturbation types x 2 = 40 rows
sampling seed = 20260702
```

Pilot overall result:

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| overall | 40 | 0.363136 | 0.021274 | 0.362603 | 0.021479 |

Pilot interpretation:

```text
The pilot suggested a moderate positive relationship between input similarity and output similarity.
However, subgroup results were unstable because each task subgroup had only 10 rows and each perturbation subgroup had only 8 rows.
```

Pilot files:

```text
outputs/rq1_pilot_input_output_similarity_rows.csv
outputs/rq1_pilot_input_output_similarity_correlations.csv
figures/attempt/rq1_pilot_input_output_similarity_scatter.png
src/25_pilot_input_output_similarity_relationship.py
```

## 5. Full 200-Row Analysis

The full analysis used all formal RQ1 item-level perturbation rows:

```text
40 original prompts x 5 perturbation types = 200 rows
```

### 5.1 Overall Result

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 0.290812 | 0.000029 | 0.262458 | 0.000174 |

Interpretation:

```text
There is a statistically significant positive relationship between input similarity and output similarity.
However, the relationship is weak-to-moderate rather than strong.
```

This means:

```text
input similarity matters, but it does not determine output similarity.
```

## 6. Results By Task Type

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| code_generation | 50 | 0.402732 | 0.003736 | 0.085618 | 0.554395 |
| factual_qa | 50 | 0.178867 | 0.213927 | 0.252355 | 0.077057 |
| math_reasoning | 50 | 0.218089 | 0.128134 | 0.097623 | 0.500023 |
| open_ended_writing | 50 | 0.379210 | 0.006611 | 0.297808 | 0.035687 |

Task-level interpretation:

```text
open_ended_writing shows the clearest task-level relationship.
code_generation has a significant Pearson correlation but weak Spearman correlation, suggesting possible outlier sensitivity or non-monotonic structure.
factual_qa and math_reasoning do not show statistically significant correlations in this analysis.
```

## 7. Results By Perturbation Type

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| context_injection | 40 | 0.229179 | 0.154880 | 0.293146 | 0.066389 |
| formatting_changes | 40 | 0.047239 | 0.772234 | 0.065116 | 0.689748 |
| paraphrasing | 40 | 0.183163 | 0.257931 | 0.382364 | 0.014895 |
| reordering | 40 | 0.137939 | 0.395990 | 0.072517 | 0.656548 |
| surface_noise | 40 | 0.204651 | 0.205248 | 0.012121 | 0.940825 |

Perturbation-level interpretation:

```text
paraphrasing shows a significant Spearman relationship, meaning that rank-order input similarity is related to rank-order output similarity for paraphrased prompts.
Other perturbation types do not show statistically significant relationships in this full analysis.
```

## 8. Main Interpretation

The full analysis supports a modest positive input-output similarity relationship:

```text
Higher input similarity is generally associated with higher output similarity.
```

However, the relationship is not strong:

```text
Pearson r = 0.290812
Spearman rho = 0.262458
```

Therefore, the correct interpretation is not:

```text
input similarity determines output similarity
```

The better interpretation is:

```text
input similarity explains part of output similarity, but task type and perturbation type still matter.
```

This helps explain why the RQ1 heatmap does not show a simple trend. Some prompt perturbations that look small at the input level can still lead to noticeable output changes, while some larger input changes may not produce large output changes.

## 9. How This Can Be Used In The Paper

Possible wording:

```text
As a secondary analysis, I examined whether the semantic similarity between original and perturbed prompts predicts the semantic similarity between their generated outputs. Across all 200 item-level perturbation cases, input similarity was positively correlated with output similarity, Pearson r = 0.291, p < .001, and Spearman rho = 0.262, p < .001. This suggests that input-level semantic preservation is related to output stability. However, the magnitude of the correlation is modest, indicating that input similarity alone does not fully explain output drift. Task type and perturbation type remain important sources of variation.
```

## 10. Files

Full-analysis data:

```text
outputs/rq1_full_input_output_similarity_rows.csv
outputs/rq1_full_input_output_similarity_correlations.csv
```

Full-analysis figure:

```text
figures/attempt/rq1_full_input_output_similarity_scatter.png
```

Full-analysis script:

```text
src/26_analyze_rq1_full_input_output_similarity.py
```

Pilot files:

```text
outputs/rq1_pilot_input_output_similarity_rows.csv
outputs/rq1_pilot_input_output_similarity_correlations.csv
figures/attempt/rq1_pilot_input_output_similarity_scatter.png
src/25_pilot_input_output_similarity_relationship.py
```
