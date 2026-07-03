# RQ1b Pilot Perturbation Results

## Purpose

RQ1b continues the full RQ1 analysis by testing whether prompt perturbations produce output changes beyond the RQ1a sampling-noise baseline.

Full RQ1:

```text
After applying a noise-baseline correction, is the ranking of the five perturbation types by their effect on output semantic similarity consistent across task types, or is the ranking task-dependent?
```

## Pilot Scope

This is a small pilot, not a formal result.

Pilot design:

```text
1 original prompt per task type
4 original prompts total
5 perturbation types per original prompt
20 perturbed prompts total
3 generated outputs per perturbed prompt
60 perturbed outputs total
```

The original prompts were selected from the existing RQ1a sampled prompt set, not newly sampled from the benchmark datasets.

## Perturbation Types

```text
paraphrasing
reordering
formatting_changes
context_injection
surface_noise
```

For this pilot, the perturbed prompts were manually constructed according to predefined perturbation categories and manually checked for semantic equivalence.

The current reported metric is Sentence-BERT cosine similarity using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Calculation

For each item and perturbation type:

```text
baseline_similarity = mean similarity among the original prompt's repeated outputs
perturbation_similarity = mean cross-similarity between original outputs and perturbed outputs
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

Interpretation:

```text
noise_corrected_drift > 0
    The perturbation produced more drift than normal sampling noise.

noise_corrected_drift <= 0
    The perturbation did not exceed the normal sampling-noise baseline in this pilot.
```

## Pilot Heatmap Table

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.041442 | -0.008114 | -0.005167 | -0.004696 |
| reordering | 0.014932 | -0.020535 | 0.024498 | 0.004064 |
| formatting_changes | -0.009633 | 0.050575 | 0.066452 | 0.000425 |
| context_injection | 0.025948 | 0.062831 | 0.005024 | 0.066072 |
| surface_noise | -0.000620 | -0.010871 | -0.012020 | 0.006697 |

## Preliminary Observations

In this small pilot:

```text
context_injection produced the largest drift for factual_qa and open_ended_writing.
formatting_changes produced the largest drift for the selected math_reasoning item.
paraphrasing produced the largest positive drift for the selected code_generation item.
surface_noise generally produced small or negative drift.
some drift values are negative, meaning the perturbed outputs were at least as similar to the original outputs as the original outputs were to each other.
```

These observations should not be treated as final findings because there is only one item per task type in this pilot.

## Important Limitations

```text
Only 1 prompt per task type.
Perturbations were manually created.
Only 3 outputs per perturbed prompt.
The result may be strongly affected by the specific selected item and wording of each perturbation.
```

Formal RQ1b analysis should use more items per task type, ideally the full RQ1a sampled set or a larger sample.

## Related Files

```text
prompts/rq1b_pilot_perturbed_prompts.csv
outputs/rq1b_pilot_perturbed_generations.csv
outputs/rq1b_pilot_perturbation_effects_by_item.csv
outputs/rq1b_pilot_perturbation_summary.csv
outputs/rq1b_pilot_heatmap_noise_corrected_drift.csv
src/11_generate_rq1b_perturbed_outputs_openai.py
src/12_analyze_rq1b_perturbation_effects.py
src/16_analyze_rq1_with_sentence_bert.py
outputs/sbert_rq1b_perturbation_effects_by_item.csv
outputs/sbert_rq1b_perturbation_summary.csv
outputs/sbert_rq1b_heatmap_noise_corrected_drift.csv
```
