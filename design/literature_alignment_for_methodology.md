# Literature Alignment For Methodology Design

## 1. Purpose

This document maps each major methodological choice in the Pioneer project to the most relevant reference papers. The goal is not to copy one paper exactly, but to make sure each part of the research design can be justified through prior work.

The project can be framed as a synthesis of six methodological ideas:

```text
1. Prompt robustness and perturbation categories
2. Task-type comparison
3. Instruction-level perturbation
4. Repeated sampling / sample-noise correction
5. Output-level semantic similarity
6. Correctness / performance-drop evaluation for objective tasks
```

## 2. Overall Literature Positioning

The project should not be presented as a direct replication of one paper. A stronger framing is:

```text
This study combines prompt-robustness evaluation with repeated-sampling noise correction.
```

In other words, the project borrows perturbation and task-comparison ideas from prompt robustness literature, but adds a sampling-noise baseline before interpreting perturbation effects.

Recommended high-level framing:

```text
Prior work has shown that LLMs are sensitive to prompt variations, including typos, paraphrases, prompt templates, and instruction perturbations. However, many studies evaluate prompt sensitivity using single generations or deterministic settings, which may confound perturbation effects with ordinary sampling variability. This project extends the prompt robustness literature by estimating a within-prompt sampling-noise baseline and then measuring noise-corrected perturbation drift.
```

## 3. Method Component Mapping

| Project design component | Main supporting literature | How the project uses it |
|---|---|---|
| Natural prompt perturbations | PromptRobust; POSIX; What Did I Do Wrong; Haase et al. | Uses paraphrasing, formatting, reordering, surface noise, and context injection as natural prompt variations |
| Character / surface noise | PromptRobust; Enhancing LLM Robustness; POSIX; Haase et al. | Uses light typos, punctuation, and capitalization noise, mainly on the instruction part |
| Paraphrasing | POSIX; What Did I Do Wrong; Haase et al. | Treats paraphrasing as a core intent-preserving perturbation |
| Formatting / template change | POSIX; Haase et al.; What Did I Do Wrong | Treats formatting changes as a separate perturbation type rather than a minor style issue |
| Information reordering | Haase et al.; POSIX template variation logic | Tests whether changing the order of prompt elements changes model outputs |
| Context injection | PromptRobust sentence-level attacks; StressTest / CheckList tradition | Uses irrelevant but non-conflicting inserted context as a stress-test perturbation |
| Instruction-level perturbation | Enhancing LLM Robustness | Perturbs the instruction template while keeping the task sample fixed where possible |
| Task-type comparison | PromptRobust; POSIX; Haase et al. | Compares prompt sensitivity across QA, math, code, and open-ended writing |
| Sampling-noise baseline | Haase et al. | Uses repeated generations to estimate within-prompt variation before interpreting perturbation effects |
| Sentence-BERT semantic similarity | Reimers & Gurevych, 2019 | Uses SBERT embeddings and cosine similarity to compare output-level semantic similarity |
| Correctness / PDR for objective tasks | PromptRobust; Enhancing LLM Robustness | Plans to use correctness or performance drop for factual QA, math, and code |
| Sensitivity / consistency framing | What Did I Do Wrong | Supports the idea that robustness can be evaluated separately from raw accuracy |

## 4. Task Type Selection

### 4.1 Factual QA

Relevant literature:

```text
PromptRobust uses SQuAD V2 for reading comprehension / QA.
POSIX uses MMLU for MCQ-style factual and knowledge tasks.
```

How this supports the project:

```text
Factual QA is a standard task type in prompt robustness evaluation because it has a constrained answer space and can support correctness evaluation.
```

Current project note:

```text
The current RQ1 design uses SQuAD V2 for factual QA. This is the stricter literature-aligned choice because PromptRobust / PromptBench include SQuAD-style reading-comprehension QA tasks, and the context-question format also supports the reordering perturbation more cleanly.
```

### 4.2 Mathematical Reasoning

Relevant literature:

```text
PromptRobust includes Mathematics / MATH as a math problem-solving task.
```

How this supports the project:

```text
Math reasoning is appropriate because it has objective final-answer correctness while still allowing variation in reasoning steps.
```

Current project note:

```text
The current RQ1 design uses MATH / Hendrycks MATH for mathematical reasoning. This replaces GSM8K because MATH is closer to the mathematics benchmark family used in the reviewed prompt-robustness literature.
```

### 4.3 Code Generation

Relevant literature:

```text
The five original prompt-robustness references do not provide a dedicated code-generation benchmark.
```

How this should be written:

```text
Code generation is an extension added by this project because code has objective functional correctness but differs from factual QA and math in output form. It can be evaluated with the same performance-drop logic used in PromptRobust and Enhancing LLM Robustness, but the dataset itself must be justified as an external addition.
```

Recommended wording:

```text
HumanEval is introduced as an external benchmark because none of the reviewed prompt-perturbation studies directly covers code generation. It is retained because code generation provides an important case where textual similarity and functional correctness may diverge.
```

### 4.4 Open-Ended Writing

Relevant literature:

```text
POSIX uses Alpaca for open-ended generation.
Haase et al. use creative generation tasks and explicitly analyze within-LLM variance.
```

How this supports the project:

```text
Open-ended writing is important because it has a high-output-space structure where sampling noise is expected to be larger. It is also the task type where output-level semantic drift is especially meaningful.
```

Current project note:

```text
The current pilot uses Alpaca, which aligns well with POSIX.
```

## 5. Perturbation Type Selection

The project's five perturbation types can be justified as follows.

| Project perturbation type | Literature support | Notes for methodology writing |
|---|---|---|
| Paraphrasing | POSIX; What Did I Do Wrong; Haase et al. | Strong support. Use as a core natural perturbation. |
| Formatting changes | POSIX prompt templates; Haase et al. formatting tweak; What Did I Do Wrong prompt rephrasings | Strong support. Treat as structural / template variation. |
| Information reordering | Haase et al. information order prompt | Strong support for open-ended prompts; can be extended carefully to other tasks. |
| Surface noise | PromptRobust character-level attacks; Enhancing LLM Robustness DeepWordBug; POSIX spelling errors; Haase et al. typo robustness | Strong support. Use light natural noise rather than full adversarial attack. |
| Context injection | PromptRobust sentence-level attacks; CheckList / StressTest style irrelevant sentence insertion | Moderate support. Use cautiously because inserted context may change task interpretation. |

Important distinction:

```text
PromptRobust and Enhancing LLM Robustness often use adversarial perturbations designed to degrade performance.
This project adapts those ideas into natural or low-intensity perturbations because the goal is to measure robustness under plausible prompt changes, not to maximize failure.
```

## 6. Perturbation Construction Method

### 6.1 What The Literature Does

| Literature | Perturbation construction |
|---|---|
| PromptRobust | Uses character-, word-, sentence-, and semantic-level adversarial attacks; also checks semantic preservation |
| Enhancing LLM Robustness | Uses character- and word-level edits to task-specific instructions |
| POSIX | Uses prompt templates, paraphrases, spelling errors, and mixtures |
| What Did I Do Wrong | Uses prompt rephrasings for classification tasks |
| Haase et al. | Uses minor phrasing change, formatting tweak, information order change, and typo robustness prompts |

### 6.2 Project Adaptation

Current pilot:

```text
Perturbations are manually constructed / rule-based.
They are designed to preserve the original task intent.
```

Why this is defensible:

```text
Manual construction gives stronger control in a small pilot because each perturbation type can be isolated. This is especially important for math and code tasks, where accidental changes to numbers, formulas, function signatures, or examples would change the task itself.
```

Future formal version:

```text
LLM-generated perturbations can be used later, but they should be manually verified for semantic equivalence.
```

## 7. Repeated Sampling And Noise Correction

Main supporting literature:

```text
Haase et al., Within-Model vs Between-Prompt Variability in Large Language Models for Creative Tasks
```

Key idea from literature:

```text
Single-sample evaluations can conflate prompt effects with within-LLM sampling variability.
```

Project adaptation:

```text
For each clean prompt and perturbed prompt, the model is sampled repeatedly.
Within-prompt output variation is estimated first.
Perturbation effects are then interpreted relative to this sample-noise baseline.
```

How to write this in methodology:

```text
Following the concern raised by Haase et al. that single-sample evaluations may confound prompt effects with within-LLM stochasticity, this study estimates a sampling-noise baseline through repeated generations before interpreting prompt-perturbation effects.
```

Important difference:

```text
Haase et al. use variance decomposition for creative tasks.
This project adapts the repeated-sampling idea to prompt perturbation and semantic drift across multiple task types.
```

## 8. Sentence-BERT Similarity Measure

Main supporting literature:

```text
Reimers and Gurevych, 2019, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
```

Why SBERT supports the project:

```text
SBERT produces semantically meaningful sentence embeddings.
The embeddings can be compared with cosine similarity.
This makes SBERT appropriate for measuring output-level semantic similarity when exact wording may differ.
```

Project use:

```text
1. Encode each generated output with Sentence-BERT.
2. Compute cosine similarity between output embeddings.
3. Use within-prompt pairwise similarity to estimate sampling-noise stability.
4. Use cross-condition similarity to estimate perturbation-induced drift.
```

Recommended wording:

```text
Output similarity is measured using Sentence-BERT (SBERT), which produces semantically meaningful sentence embeddings that can be compared using cosine similarity. This is appropriate for the present study because two LLM outputs may express similar content with different surface wording.
```

Important limitation:

```text
SBERT measures semantic similarity, not task correctness.
Therefore, factual QA, math reasoning, and code generation should also include correctness-based evaluation in RQ2.
```

## 9. RQ1 Alignment

RQ1:

```text
After applying a sampling-noise correction, are perturbation effects consistent across task types, or task-dependent?
```

Best supporting references:

| RQ1 component | Supporting literature |
|---|---|
| Prompt sensitivity is task-dependent | POSIX |
| Perturbation categories | PromptRobust; POSIX; Haase et al. |
| Need to separate sampling noise | Haase et al. |
| Semantic output comparison | Sentence-BERT |

Most important comparison:

```text
POSIX finds that different task formats show different sensitivity patterns: prompt templates are especially sensitive for MCQ-style tasks, while paraphrasing is highly sensitive for open-ended generation.
The current project tests a related pattern-level question using output-level semantic drift and sampling-noise correction.
```

Do not claim:

```text
The project directly replicates POSIX.
```

Better claim:

```text
The project extends the task-dependent prompt sensitivity question raised by POSIX by using output-level semantic drift and adding a sampling-noise correction.
```

## 10. RQ2 Alignment

RQ2 should connect semantic drift to correctness or task performance.

Best supporting references:

| RQ2 component | Supporting literature |
|---|---|
| Correctness / performance drop | PromptRobust; Enhancing LLM Robustness |
| Sensitivity independent of accuracy | What Did I Do Wrong |
| Semantic similarity alone is insufficient | Enhancing LLM Robustness; SBERT limitation |
| Need repeated sampling before interpreting changes | Haase et al. |

Recommended RQ2 framing:

```text
RQ2 asks whether output-level semantic drift corresponds to correctness change in objectively evaluable tasks. For factual QA and math reasoning, correctness can be evaluated using reference answers. For code generation, correctness can be evaluated using functional tests. This follows the performance-drop logic used in PromptRobust and Enhancing LLM Robustness, while using repeated sampling to avoid overinterpreting single-generation failures.
```

Open-ended writing:

```text
Open-ended writing should not be included in automatic correctness analysis unless a separate human or rubric-based quality evaluation is developed.
```

## 11. What Is Fully Literature-Supported vs Project Extension

| Design choice | Status |
|---|---|
| Factual QA as a task type | Fully supported |
| Math reasoning as a task type | Fully supported |
| Open-ended writing / Alpaca | Fully supported by POSIX |
| Code generation / HumanEval | Project extension; dataset not from current references |
| Paraphrasing | Fully supported |
| Formatting / template changes | Fully supported |
| Information reordering | Supported mainly by Haase et al. |
| Surface noise | Fully supported |
| Context injection | Supported, but should be treated as stress-test / cautious perturbation |
| Repeated sampling | Fully supported by Haase et al. |
| Noise-corrected perturbation drift | Project extension built from Haase et al. + prompt robustness literature |
| Sentence-BERT cosine similarity | Fully supported as semantic similarity method |
| PDR / correctness drop | Fully supported for objective tasks |
| Code pass-rate PDR | Project extension of PDR logic |

## 12. Recommended Methodology Paragraph

The following paragraph can be adapted into the paper draft:

```text
The methodology combines prior work on prompt robustness with repeated-sampling evaluation. PromptRobust and Enhancing LLM Robustness motivate the use of prompt perturbations such as character-level noise, word-level edits, and instruction-level perturbations, while POSIX and What Did I Do Wrong motivate treating prompt sensitivity as a property that can vary across prompt formulations and task types. Haase et al. show that single-sample LLM evaluations may confound prompt effects with within-model sampling variability; therefore, this study estimates a sampling-noise baseline through repeated generations before interpreting perturbation-induced drift. Output-level semantic similarity is measured using Sentence-BERT cosine similarity, following Reimers and Gurevych's SBERT framework for semantically meaningful sentence embeddings. For objectively evaluable tasks, RQ2 will additionally use correctness-based measures adapted from the performance-drop logic in PromptRobust and Enhancing LLM Robustness.
```
