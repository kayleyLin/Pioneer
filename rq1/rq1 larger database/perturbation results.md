# RQ1 Larger Database: Perturbation Results

Date analyzed: 2026-07-03

This document records the expanded RQ1 perturbation results using 50 prompts per task type.

## 1. Data Scale

| Task type | Perturbed prompts | Outputs per perturbed prompt | Total perturbed outputs |
|---|---:|---:|---:|
| factual_qa | 250 | 5 | 1250 |
| math_reasoning | 250 | 5 | 1250 |
| code_generation | 250 | 5 | 1250 |
| open_ended_writing | 250 | 5 | 1250 |

Total perturbed prompts: 1000

Total perturbed outputs: 5000

Each task type includes 50 original prompts and five perturbation types:

```text
paraphrasing
reordering
formatting_changes
context_injection
surface_noise
```

## 2. Noise-Corrected Drift

Noise-corrected drift is calculated as:

```text
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

where:

```text
baseline_similarity = within-prompt similarity among repeated original outputs
perturbation_similarity = similarity between original outputs and perturbed outputs
```

Positive values indicate that the perturbed outputs are less similar to the original-output baseline. Values near zero indicate that the perturbation effect is close to ordinary sampling noise.

## 3. Mean Noise-Corrected Drift Heatmap Values

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.016026 | 0.186192 | 0.033969 | 0.135086 |
| reordering | 0.006532 | 0.043968 | -0.002101 | 0.030975 |
| formatting_changes | 0.002906 | 0.012512 | -0.001951 | 0.009171 |
| context_injection | 0.004053 | 0.004940 | 0.000989 | 0.006349 |
| surface_noise | 0.003543 | 0.001818 | -0.001162 | 0.000020 |

Task-level pattern:

```text
Paraphrasing has the largest noise-corrected drift in all four task types.
Factual QA and open-ended writing show the largest paraphrasing effects.
Reordering is the second-largest perturbation for factual QA and open-ended writing.
Math reasoning shows relatively small or near-zero corrected drift for all non-paraphrasing perturbations.
Surface noise and context injection generally remain close to zero after baseline correction.
```

## 4. Friedman Tests

The Friedman test compares perturbation types within each task type using a repeated-measures design.

| Task type | n items | Friedman statistic | p value | Interpretation |
|---|---:|---:|---:|---|
| code_generation | 50 | 9.504000 | 0.049665 | Significant at alpha = .05, but weak relative to other tasks |
| factual_qa | 50 | 112.526203 | < .001 | Significant perturbation-type differences |
| math_reasoning | 50 | 37.568000 | 1.37571e-07 | Significant perturbation-type differences |
| open_ended_writing | 50 | 38.830835 | 7.55e-08 | Significant perturbation-type differences |

## 5. Significant Holm-Corrected Pairwise Comparisons

Only significant pairwise Wilcoxon comparisons are listed here.

| Task type | Perturbation 1 | Perturbation 2 | Mean difference | Holm-adjusted p | Significant |
|---|---|---|---:|---:|---|
| factual_qa | paraphrasing | reordering | 0.142224 | 6.2e-11 | Yes |
| factual_qa | paraphrasing | formatting_changes | 0.173679 | 3.4e-11 | Yes |
| factual_qa | paraphrasing | context_injection | 0.181252 | < .001 | Yes |
| factual_qa | paraphrasing | surface_noise | 0.184373 | < .001 | Yes |
| factual_qa | reordering | formatting_changes | 0.031455 | 0.001850 | Yes |
| factual_qa | reordering | context_injection | 0.039028 | 1.7605063e-05 | Yes |
| factual_qa | reordering | surface_noise | 0.042149 | 2.763281e-06 | Yes |
| factual_qa | formatting_changes | surface_noise | 0.010694 | 0.014456 | Yes |
| math_reasoning | paraphrasing | reordering | 0.036070 | 5.3101e-08 | Yes |
| math_reasoning | paraphrasing | formatting_changes | 0.035921 | 9.060987e-06 | Yes |
| math_reasoning | paraphrasing | context_injection | 0.032981 | 0.000310 | Yes |
| math_reasoning | paraphrasing | surface_noise | 0.035131 | 2.931042e-06 | Yes |
| open_ended_writing | paraphrasing | reordering | 0.104111 | 0.005713 | Yes |
| open_ended_writing | paraphrasing | formatting_changes | 0.125915 | 2.6159112e-05 | Yes |
| open_ended_writing | paraphrasing | context_injection | 0.128737 | 0.000202 | Yes |
| open_ended_writing | paraphrasing | surface_noise | 0.135066 | 2.283073e-06 | Yes |
| open_ended_writing | reordering | surface_noise | 0.030954 | 0.037175 | Yes |

## 6. Main Interpretation

The expanded n=50 perturbation results support a task-dependent prompt sensitivity pattern.

The most robust pattern is that paraphrasing produces the largest noise-corrected semantic drift across all task types. However, the magnitude of this effect differs by task type. The effect is strongest for factual QA and open-ended writing, smaller for math reasoning, and smallest for code generation.

The smaller perturbations, especially context injection and surface noise, are often close to zero after baseline correction. This means that their apparent uncorrected effect is largely explained by ordinary LLM sampling noise rather than by the prompt perturbation itself.

## 7. Generated Files

Analysis outputs:

```text
outputs/sbert_rq1_n50_perturbation_effects_by_item.csv
outputs/sbert_rq1_n50_perturbation_summary.csv
outputs/sbert_rq1_n50_heatmap_noise_corrected_drift.csv
outputs/sbert_rq1_n50_uncorrected_perturbation_summary.csv
outputs/sbert_rq1_n50_uncorrected_heatmap_drift.csv
outputs/rq1_n50_perturbation_friedman.csv
outputs/rq1_n50_perturbation_pairwise_wilcoxon.csv
```

Figures:

```text
figures/rq1_n50_uncorrected_drift_heatmap.png
figures/rq1_n50_noise_corrected_drift_heatmap.png
```

Scripts:

```text
src/32_analyze_rq1_n50_perturbations_sbert.py
src/33_analyze_rq1_n50_perturbation_significance.py
src/34_create_rq1_n50_heatmaps.py
```
