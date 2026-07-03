# RQ1 Limitations Notes

## 1. Purpose

This document records methodological limitations identified during the RQ1 formal-pipeline preparation. It is intended as a working notes file for later use in the methodology limitations section.

## 2. Limitation: Exact-Match Check For Reordering Validation

For the reordering perturbation, one simple validation idea is to compare the original prompt and the perturbed prompt token by token.

The basic logic is:

```text
1. Tokenize the original prompt.
2. Tokenize the reordered prompt.
3. Compare tokens at the same positions.
4. Count how many positions contain the same token.
5. If the number of matching positions equals the number of original tokens, the reordered prompt is unchanged.
6. If unchanged, remove, revise, or replace that prompt before running output generation.
```

A cleaner implementation is a normalized exact-match check:

```text
normalize(original_prompt) == normalize(reordered_prompt)
```

where normalization can include:

```text
lowercasing
removing extra whitespace
optionally standardizing punctuation
```

## 3. What This Check Can Do

This check is useful as a minimum validity filter.

It can detect cases where:

```text
the reordering perturbation did not change the prompt at all
the original and reordered prompts are identical after normalization
```

This is important because a reordering perturbation that is identical to the original prompt should not be included as a valid perturbed prompt.

## 4. Main Limitation

The exact-match check does not prove that meaningful reordering occurred.

It only proves that the two prompts are not exactly the same.

For example, it may fail to catch weak or invalid perturbations such as:

```text
only one punctuation mark changed
only one word was replaced
only spacing or capitalization changed
the sentence was lightly paraphrased but not reordered
the order changed grammatically but not at the level of information units
the prompt changed in a way that affects meaning rather than only order
```

Therefore, the exact-match check should not be described as a complete reordering-validation algorithm.

## 5. How To Describe This In The Methodology

Recommended wording:

```text
Before output generation, all perturbed prompts were screened to remove unchanged perturbations. For reordering, a normalized exact-match check was used as a minimum validity check to ensure that the reordered prompt was not identical to the original prompt. Prompts that failed this check were manually revised or replaced.
```

Important caveat:

```text
This check ensures that the perturbation is not identical to the original prompt, but it does not quantify the degree of information-order change.
```

## 6. Possible Future Improvement

A stronger future version could add an order-distance metric, such as Kendall tau distance, to compare the order of shared information units between the original and reordered prompts.

Potential future logic:

```text
1. Segment the prompt into reorderable information units.
2. Match shared units between the original and reordered prompt.
3. Compare their order using Kendall tau or another ranking-distance metric.
4. Accept the reordering only if semantic similarity remains high and order distance exceeds a minimum threshold.
```

Candidate supporting literature to review:

| Reference | Potential use |
|---|---|
| Lapata (2006), "Automatic Evaluation of Information Ordering: Kendall's Tau" | Main candidate for information-ordering validation |
| Kendall (1938), "A New Measure of Rank Correlation" | Original rank-correlation method |
| Kumar and Vassilvitskii (2010), "Generalized Distances between Rankings" | Broader ranking-distance framework |

## 7. Current Practical Decision

For the current RQ1 pipeline, the normalized exact-match check can be used as a simple screening rule.

However, because it is only a minimum check, reordering prompts that pass the exact-match check may still require manual review for:

```text
semantic equivalence
actual information-order change
naturalness
absence of added hints or changed task constraints
```

