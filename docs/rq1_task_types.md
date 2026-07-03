# RQ1 Task Type Design

## Current Scope

RQ1 focuses on sampling-noise-corrected prompt sensitivity:

```text
After estimating ordinary sampling noise from repeated generations of the same prompt, do different perturbation types produce different levels of output semantic drift across task types?
```

The current RQ1 design therefore includes both the repeated-generation baseline and the perturbed-prompt comparison.

## Task Type Classification

This project classifies tasks by expected output structure and evaluation style, not only by benchmark name.

| Task type | Benchmark source | Output structure | Why it is included |
|---|---|---|---|
| factual_qa | SQuAD V2 | Context-question factual answer | Tests factual response stability with a literature-aligned QA dataset and reorderable context/question structure |
| math_reasoning | MATH / Hendrycks MATH | Mathematical reasoning with a final answer | Tests whether reasoning traces vary across repeated generations using a literature-aligned math dataset |
| code_generation | HumanEvalPack Python | Function implementation | Tests variation in code form for the same programming task |
| open_ended_writing | Alpaca | Open-ended instruction response | Tests high-output-space generation where sampling noise is expected to be larger |

## Sampling Rule

For the formal RQ1 experiment, prompts are sampled separately within each task type using a fixed random seed.

```text
sampling method = stratified random sampling by task type
random seed = 20260623
sample size = 10 prompts per task type
total sample size = 40 original prompts
```

Using a fixed random seed makes the formal sample reproducible: the same script should select the same items again.

## RQ1 Measurement

For each sampled prompt:

```text
1. Send the exact same prompt to the same LLM multiple times.
2. Keep model, temperature, top_p, and system prompt fixed.
3. Compute pairwise similarity among the repeated outputs.
4. Average the pairwise similarities to estimate within-prompt stability.
5. Convert this into sampling-noise drift as 1 - mean_similarity.
```

The final RQ1 comparison is task-level:

```text
mean sampling-noise drift for factual_qa
mean sampling-noise drift for math_reasoning
mean sampling-noise drift for code_generation
mean sampling-noise drift for open_ended_writing
```
