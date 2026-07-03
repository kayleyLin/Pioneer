# Research Proposal Feedback Memory

## Overall Assessment

The proposal has a strong and viable core idea: prompt perturbation effects should be separated from the model's own sampling noise. This is more valuable than simply showing that LLMs are prompt-sensitive, because it frames the problem as a measurable empirical methodology.

## Strengths

- The research gap is clear. The literature review connects adversarial perturbation, natural rephrasing, task-dependent sensitivity, deterministic versus non-deterministic generation, and sampling noise.
- The research questions are well layered. RQ1 examines noise-corrected perturbation rankings, RQ2 studies the relationship between semantic drift and correctness change, and RQ3 explores cross-model consistency.
- The methodology is feasible. For the current RQ1 design, factual QA has been updated to SQuAD V2 and math reasoning has been updated to MATH / Hendrycks MATH for stronger alignment with the prompt-robustness literature. Combining semantic similarity with correctness remains stronger than using embedding similarity alone.

## Main Risks And Issues To Fix

### 1. Make The Noise Baseline More Precise

The proposal currently says that multiple generations from the same prompt form a noise baseline, but the exact computation should be specified.

Suggested definition:

```text
noise = mean pairwise similarity among repeated outputs from the same prompt
perturbation_similarity = mean pairwise similarity between outputs from the original prompt and outputs from the perturbed prompt
noise_corrected_drift = noise - perturbation_similarity
```

It should also clarify whether the baseline uses only the original prompt's within-output similarity, or both the original and perturbed prompt versions' within-output similarities. A stronger approach is to include both, because the perturbed prompt also has sampling noise.

### 2. Increase Or Justify The Number Of Repeated Samples

The planned 3 to 5 generations per prompt version may be too small for a stable noise baseline. If resources allow, use 5 samples for the main open-source model and possibly 3 samples for commercial API models.

This should also be framed as an exploratory study if the sample count remains small.

### 3. Improve The Statistical Analysis Plan

The phrase "repeated-measures ANOVA, with item as the within-subjects factor" is not quite accurate.

A better framing:

- Perturbation type is a within-item factor.
- Task type is a between-item factor.
- Item should be treated as a repeated unit or random effect.

Recommendation: make a linear mixed-effects model the primary analysis method rather than only a robustness check, because the data are naturally nested: outputs within prompt versions, prompt versions within items, and items within task types.

### 4. Define Correctness Change More Clearly

RQ2 needs a more precise definition of "correctness changed."

Suggested definition:

```text
For each item, correctness is evaluated separately under the original prompt and the perturbed prompt.
correctness_changed = 1 if the correctness label differs between the two prompt versions, and 0 otherwise.
```

It may also help to distinguish between:

- correct -> incorrect
- incorrect -> correct

These two transitions have different interpretations.

### 5. Acknowledge Limits Of Sentence-BERT For Math And Code

Sentence-BERT similarity may not capture important correctness differences in math and code. A small token-level change can make an answer wrong even if the embedding similarity remains high.

This is not necessarily a flaw. It can support RQ2 by testing whether semantic drift actually predicts correctness change. However, the proposal should state clearly that Sentence-BERT is a proxy for output similarity, not a proxy for correctness.

### 6. Tighten The Definition Of Context Injection

"Context injection" currently means adding unrelated background information. This may be challenged because unrelated context can subtly change the prompt's meaning or pragmatic intent.

Suggested wording:

```text
irrelevant but non-conflicting context
```

All context-injected prompts should be manually checked to ensure they do not alter the task, constraints, or expected answer.

## Highest-Priority Revisions

1. Rewrite the noise baseline section with a clear formula or step-by-step procedure.
2. Make the linear mixed-effects model the main statistical analysis, or clearly justify why ANOVA is sufficient.
3. Define correctness change precisely.
4. Add an estimated experiment size, for example:

```text
65 items x 6 prompt versions x 5 samples x 2 models = 3900 generations
```

5. Revise "context injection" to "irrelevant but non-conflicting context."

## One-Sentence Summary

The proposal is promising because it has a clear gap, feasible empirical design, and a meaningful methodological contribution; the main improvement is to make the noise-corrected sensitivity metric and statistical testing procedure fully explicit and reproducible.
