# RQ1 Input Similarity And Output Similarity

## 1. Purpose

This document explores a new analysis direction:

```text
Are more similar perturbed prompts also associated with more similar model outputs?
```

This is motivated by the current RQ1 heatmap results, where the perturbation effects are not uniformly patterned across task types. Instead of only comparing perturbation categories, this analysis asks whether the magnitude of the input-level change predicts the magnitude of the output-level change.

## 2. Pilot Design

This pilot uses existing formal RQ1 data only. It does not call the OpenAI API.

Sampling:

```text
source data = 200 item-level perturbation rows
pilot sample = 2 rows per task_type x perturbation_type cell
task types = 4
perturbation types = 5
pilot rows = 4 x 5 x 2 = 40
sampling seed = 20260702
```

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

## 3. Pilot Correlation Results

Overall relationship:

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| overall | 40 | 0.363136 | 0.021274 | 0.362603 | 0.021479 |

By task type:

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| code_generation | 10 | 0.827923 | 0.003099 | 0.078788 | 0.828717 |
| factual_qa | 10 | 0.319812 | 0.367681 | 0.325178 | 0.359238 |
| math_reasoning | 10 | 0.479036 | 0.161275 | 0.381818 | 0.276255 |
| open_ended_writing | 10 | 0.362469 | 0.303318 | 0.386532 | 0.269869 |

By perturbation type:

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| context_injection | 8 | 0.373797 | 0.361680 | 0.333333 | 0.419753 |
| formatting_changes | 8 | -0.016572 | 0.968934 | -0.191620 | 0.649410 |
| paraphrasing | 8 | 0.000877 | 0.998355 | 0.452381 | 0.260405 |
| reordering | 8 | 0.538216 | 0.168794 | 0.238095 | 0.570156 |
| surface_noise | 8 | -0.193893 | 0.645460 | -0.275454 | 0.509054 |

## 4. Preliminary Interpretation

The overall pilot shows a modest positive relationship:

```text
higher input similarity is associated with higher output similarity
```

The overall correlation is statistically significant in this small pilot:

```text
Pearson r = 0.363136, p = 0.021274
Spearman rho = 0.362603, p = 0.021479
```

However, the subgroup results are not stable enough to interpret strongly. Each task subgroup has only 10 rows, and each perturbation subgroup has only 8 rows. There are visible outliers in the scatter plot, so the pilot should be treated as directional evidence rather than a final result.

Important nuance:

```text
Input similarity alone does not fully explain output similarity.
Task type and perturbation type still appear to matter.
```

This means the next analysis should likely use the full 200 item-level perturbation rows rather than only this 40-row pilot.

## 5. Generated Files

Pilot data:

```text
outputs/rq1_pilot_input_output_similarity_rows.csv
outputs/rq1_pilot_input_output_similarity_correlations.csv
```

Pilot figure:

```text
figures/attempt/rq1_pilot_input_output_similarity_scatter.png
```

Pilot script:

```text
src/25_pilot_input_output_similarity_relationship.py
```

## 6. Recommended Next Step

The pilot suggested that the full 200-row analysis was worth running. The full analysis has now been completed and is recorded below.

## 7. Full 200-Row Analysis

The full analysis uses all item-level perturbation rows:

```text
40 original prompts x 5 perturbation types = 200 item-level perturbation rows
```

It uses the same definitions as the pilot:

```text
input_similarity = SBERT(original_prompt, perturbed_prompt)
output_similarity = average SBERT(original_outputs, perturbed_outputs)
```

### 7.1 Overall Result

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 0.290812 | 0.000029 | 0.262458 | 0.000174 |

Interpretation:

```text
There is a statistically significant positive relationship between input similarity and output similarity.
However, the relationship is modest rather than strong.
```

This means that prompts that remain more similar after perturbation tend to produce more similar outputs, but input similarity alone does not fully explain output behavior.

### 7.2 Correlation By Task Type

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| code_generation | 50 | 0.402732 | 0.003736 | 0.085618 | 0.554395 |
| factual_qa | 50 | 0.178867 | 0.213927 | 0.252355 | 0.077057 |
| math_reasoning | 50 | 0.218089 | 0.128134 | 0.097623 | 0.500023 |
| open_ended_writing | 50 | 0.379210 | 0.006611 | 0.297808 | 0.035687 |

Task-level interpretation:

```text
The clearest task-level relationship appears in open-ended writing.
Code generation has a significant Pearson correlation, but its Spearman correlation is weak, suggesting sensitivity to outliers or non-monotonic structure.
Factual QA and math reasoning do not show statistically significant correlations in this analysis.
```

### 7.3 Correlation By Perturbation Type

| Group | n | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|---:|
| context_injection | 40 | 0.229179 | 0.154880 | 0.293146 | 0.066389 |
| formatting_changes | 40 | 0.047239 | 0.772234 | 0.065116 | 0.689748 |
| paraphrasing | 40 | 0.183163 | 0.257931 | 0.382364 | 0.014895 |
| reordering | 40 | 0.137939 | 0.395990 | 0.072517 | 0.656548 |
| surface_noise | 40 | 0.204651 | 0.205248 | 0.012121 | 0.940825 |

Perturbation-level interpretation:

```text
Paraphrasing shows a significant Spearman relationship, suggesting that rank-order input similarity matters for output similarity in paraphrased prompts.
Other perturbation types do not show statistically significant relationships in this full analysis.
```

### 7.4 Interpretation For RQ1

This analysis helps explain why the heatmap does not show a simple trend. There is some relationship between input similarity and output similarity, but it is not strong enough to explain all output drift.

The result suggests:

```text
Input similarity matters, but it is only one factor.
Task type and perturbation type still shape output behavior.
Some low-output-similarity cases occur even when input similarity is high.
Some high-output-similarity cases occur even when input similarity is lower.
```

This supports adding input-output similarity as a secondary explanatory analysis rather than replacing the current RQ1 perturbation heatmap.

## 8. Full Analysis Files

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
