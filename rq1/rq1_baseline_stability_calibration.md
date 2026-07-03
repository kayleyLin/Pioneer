# RQ1 Baseline Stability Calibration

## Purpose

This calibration checks whether the sampling-noise baseline changes substantially when the number of repeated generations per prompt increases.

Question:

```text
Are 3 repeated generations per prompt enough to estimate the sampling-noise baseline, or does the baseline change substantially when using 5, 7, or 10 generations?
```

This is not a separate research question. It is a methodological calibration step for RQ1.

The current reported metric is Sentence-BERT cosine similarity using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Calibration Design

Calibration set:

```text
2 prompts per task type
4 task types
8 prompts total
```

Generation count:

```text
10 outputs per prompt
80 outputs total
```

Then the baseline was recalculated using:

```text
n = 3
n = 5
n = 7
n = 10
```

Because 10 outputs were generated for each prompt, smaller n values were estimated by using the first n outputs.

## Task-Level Results

| Task type | n=3 | n=5 | n=7 | n=10 |
|---|---:|---:|---:|---:|
| code_generation | 0.073214 | 0.065846 | 0.062641 | 0.058804 |
| factual_qa | 0.041099 | 0.035543 | 0.031909 | 0.031575 |
| math_reasoning | 0.077373 | 0.063475 | 0.061063 | 0.060591 |
| open_ended_writing | 0.178248 | 0.169233 | 0.165099 | 0.168513 |

## Interpretation

The Sentence-BERT baseline estimates do not change dramatically across n=3, n=5, n=7, and n=10 in this small calibration. The general task-level pattern remains similar:

```text
open_ended_writing has the highest sampling-noise drift.
factual_qa has relatively low sampling-noise drift.
code_generation and math_reasoning fall between these cases in this calibration.
```

The change from n=5 to n=10 is generally modest. This suggests that n=5 may be a reasonable balance between stability and API cost for the formal experiment. However, n=3 is still useful for pilot testing because it gives a similar broad pattern with lower cost.

## Recommended Use

Recommended plan:

```text
pilot experiments: n = 3 generations per prompt
formal RQ1 experiment, if budget allows: n = 5 generations per prompt
```

If resources are limited, n=3 can be retained with a clear limitation. If the goal is a stronger formal analysis, n=5 is preferable because it uses 10 pairwise comparisons per prompt instead of only 3.

Pair counts:

```text
n = 3  -> 3 pairwise comparisons
n = 5  -> 10 pairwise comparisons
n = 7  -> 21 pairwise comparisons
n = 10 -> 45 pairwise comparisons
```

## Limitations

```text
Only 2 prompts per task type were used in the calibration.
The calibration uses the first n outputs rather than repeated random subsampling of n outputs from the 10 generations.
```

Formal improvement:

```text
Optionally use bootstrap/random subsets from the 10 outputs to estimate variability more robustly.
```

## Related Files

```text
prompts/rq1_calibration_prompts.csv
outputs/rq1_calibration_generations.csv
outputs/rq1_baseline_stability_by_item.csv
outputs/rq1_baseline_stability_by_task.csv
outputs/sbert_rq1_baseline_stability_by_item.csv
outputs/sbert_rq1_baseline_stability_by_task.csv
src/13_create_rq1_calibration_set.py
src/14_generate_rq1_calibration_outputs_openai.py
src/15_analyze_rq1_baseline_stability.py
src/16_analyze_rq1_with_sentence_bert.py
```
