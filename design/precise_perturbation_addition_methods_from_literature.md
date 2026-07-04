# Precise Perturbation Addition Methods Used In This Study

## Purpose

This document records only the perturbation-addition methods actually used in the Pioneer experiment. It is intended to support the paper's `Perturbation Construction` / `Experiments` section.

It does **not** list every perturbation method found in the literature. Earlier broad mapping files already do that. This file keeps only the final methods adopted by this study.

## Writing Frame

The study does not directly replicate one prior perturbation pipeline. Instead, each perturbation type uses the closest corresponding operational method from prior work:

```text
paraphrasing -> POSIX-style GPT-3.5-Turbo paraphrase generation
reordering -> Haase et al. information-order variation
formatting_changes -> POSIX-style prompt-template variation
context_injection -> PromptRobust-style irrelevant sentence insertion, softened for natural context
surface_noise -> POSIX-style spelling-error operations
```

For all five perturbation types, the project applies only one perturbation type at a time and preserves task-critical content.

## Final Perturbation Addition Table

| Project perturbation type | How this study adds the perturbation | Literature method followed | What is changed | What must stay unchanged |
|---|---|---|---|---|
| paraphrasing | The original prompt or main instruction is rewritten with GPT-3.5-Turbo under a meaning-preservation instruction. The generated paraphrase is checked against the original prompt. If it changes the answer target, task constraints, numbers, entities, examples, or code signature, it is rejected and regenerated. | POSIX uses GPT-3.5-Turbo to generate paraphrases that preserve the original intent and meaning. | Wording and sentence phrasing | Task intent, answer target, constraints, entities, numbers, examples, code signatures |
| reordering | The prompt is separated into components such as instruction, context, question, constraints, examples, and output requirement. These components are rearranged into a different order while keeping the same information. | Haase et al. use an information-order prompt variation in which the same information is presented in a different order. | Order of prompt components | All original information; math conditions; factual evidence; code signatures; examples |
| formatting_changes | The prompt content is converted into a different presentation format, such as bullet points, numbered fields, labels, separators, capitalization changes, or answer markers. | POSIX uses prompt-template variations that alter structure, labels, separators, capitalization, and answer markers while preserving meaning. Haase et al. also use formatting-tweak prompt variations. | Layout, labels, separators, answer markers, list structure | Semantic content, task requirements, answer target |
| context_injection | One neutral, irrelevant, non-conflicting sentence is inserted before or after the main task instruction. The inserted sentence must not provide evidence, hints, examples, assumptions, or new task conditions. | PromptRobust / PromptBench use sentence-level perturbations, including irrelevant or extraneous sentence insertion. This study adapts that idea into a milder natural-context version. | Surrounding background context | Evidence, answer hints, assumptions, examples, facts, task conditions |
| surface_noise | Minor spelling, punctuation, spacing, insertion, omission, transposition, or substitution errors are added to non-critical instruction words only. | POSIX uses spelling-error operations: insertion, omission, transposition, and substitution. PromptRobust and Agrawal et al. provide secondary support for character-level perturbations. | Low-level surface form of non-critical instruction words | Numbers, formulas, named entities, code signatures, examples, answer-critical words |

## Short Paper Paragraph

Use this paragraph if the paper needs prose instead of a table:

```text
Perturbations were constructed using a component-level adaptation of prior perturbation methods. Paraphrasing followed POSIX by using GPT-3.5-Turbo to generate intent-preserving rephrasings. Reordering followed Haase et al.'s information-order variation by rearranging prompt components while keeping the same information. Formatting changes followed POSIX-style prompt-template variation by changing labels, separators, bullets, or answer markers without changing content. Context injection adapted PromptRobust sentence-level irrelevant-context insertion by adding one neutral, non-conflicting background sentence. Surface noise followed POSIX spelling-error operations, including insertion, omission, transposition, and substitution, with support from PromptRobust and Agrawal et al.'s character-level perturbations. For all perturbation types, task-critical information such as numbers, entities, formulas, code signatures, examples, and answer targets was preserved.
```

## Why Not Use Other Literature Methods Directly

The following literature methods are not part of the final experimental perturbation set:

| Literature method not used directly | Reason |
|---|---|
| TextFooler / BertAttack word replacement | Automatic synonym replacement can change facts, math conditions, or code requirements. |
| CheckList random alphanumeric strings | Too artificial for this study's natural everyday prompt-variation framing. |
| StressTest repeated logic strings | Too adversarial/artificial; the project instead uses one mild irrelevant sentence. |
| PromptRobust semantic-level cross-lingual style | Does not match the current five perturbation categories. |
| POSIX mixture condition | Mixes multiple perturbation sources; this study isolates one perturbation type at a time. |

## Citation Targets

Use these as the main citations for the construction procedure:

| Project perturbation | Primary citation | Secondary citation if needed |
|---|---|---|
| paraphrasing | POSIX | What Did I Do Wrong?; Haase et al. |
| reordering | Haase et al. | POSIX template-structure logic |
| formatting_changes | POSIX | Haase et al. |
| context_injection | PromptRobust / PromptBench | CheckList / StressTest tradition, but only as background |
| surface_noise | POSIX | PromptRobust / PromptBench; Agrawal et al. |

## Key Claim To Use

```text
The final perturbation construction is hybrid but fixed: each perturbation type is assigned one primary addition method before the main experiment, and that method is applied consistently across task types.
```
