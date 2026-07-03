# Component-Level Literature Mapping

## 1. Purpose

This document maps each part of the Pioneer research design to the closest method used in the reference literature. The goal is not to force the entire project to match one paper. Instead, each component can be justified separately:

```text
task type -> closest task/dataset in a reference paper
perturbation type -> closest perturbation method in a reference paper
evaluation method -> closest metric or evaluation logic in a reference paper
sampling/noise correction -> closest repeated-sampling method in a reference paper
```

This is the most appropriate framing for the project because the study combines several existing ideas into one noise-corrected prompt-sensitivity design.

## 2. Recommended Writing Logic

The paper should not say:

```text
This study follows Paper X.
```

A stronger and more accurate statement is:

```text
This study synthesizes methods from prior prompt-robustness and LLM-variability studies. Each task type, perturbation category, and evaluation criterion is selected based on the closest corresponding method in the literature, while the sample-noise correction is introduced to make perturbation comparisons more reliable.
```

## 3. Task-Level Mapping

| Project task type | Current / planned dataset | Closest literature match | Reference paper | Match level | How to describe it |
|---|---|---|---|---|---|
| Factual QA | SQuAD V2 | Reading comprehension / factual QA with SQuAD V2 | PromptRobust / PromptBench | Direct / high | Factual QA now follows the reading-comprehension / QA dataset family used in the prompt-robustness literature. The context-question structure also supports the reordering perturbation more cleanly than single-question factual datasets. |
| Math reasoning | MATH / Hendrycks MATH | Math problem-solving / Mathematics dataset | PromptRobust / PromptBench | Direct / high | Math reasoning now uses a literature-aligned mathematics benchmark rather than GSM8K. This keeps the task objectively evaluable while matching the benchmark family used in the reviewed robustness work. |
| Code generation | HumanEval / HumanEvalPack | No exact match in the six current references | Project extension | Extension | Code generation is added because it is objectively evaluable through functional correctness, but it should be acknowledged as an extension beyond the reviewed prompt-robustness datasets. |
| Open-ended writing | Alpaca | Open-ended generation with Alpaca | POSIX | Direct / high | Open-ended writing can be directly connected to POSIX, which uses Alpaca for open-ended generation. |
| Open-ended creative generation | Optional AUT-style task | Alternative Uses Task | Haase et al. | Optional match | If the project shifts toward creativity/open-ended idea generation, Haase et al.'s AUT task is a close reference. |

## 4. Perturbation-Level Mapping

This table identifies which perturbation methods can be copied or adapted from which papers.

| Project perturbation | Exact / closest literature method | Reference paper | Match level | Can copy method directly? | Project adaptation |
|---|---|---|---|---:|---|
| Paraphrasing | GPT-3.5-Turbo generated paraphrases preserving intent | POSIX | Very high | Yes, if using LLM-generated paraphrases | Generate paraphrases with an LLM, then manually verify semantic equivalence. |
| Paraphrasing | LLM rephrases task description by changing length or adding unnecessary words while preserving meaning | What Did I Do Wrong? | Very high | Yes | Use for instruction/task-description paraphrasing. |
| Paraphrasing | Minor phrasing change | Haase et al. | High | Yes | Use lightweight manual rewording for pilot trials. |
| Formatting changes | Prompt template variations | POSIX | Very high | Yes | Change labels, separators, capitalization, answer markers, or prompt layout while preserving content. |
| Formatting changes | Formatting tweak / structural constraint | Haase et al. | High | Yes | Use bullet lists, numbered fields, explicit output fields, or structure constraints. |
| Information reordering | Information order prompt | Haase et al. | Direct | Yes | Reorder prompt elements while keeping the same information. |
| Surface noise | Spelling errors with insertion, omission, transposition, substitution | POSIX | Direct | Yes | Use POSIX's four spelling-error operations, but apply only to non-critical instruction words. |
| Surface noise | Character-level TextBugger / DeepWordBug | PromptRobust | High | Partly | Use as support for typo/noise perturbation, but weaken from adversarial attack to natural surface noise. |
| Surface noise | DeepWordBug on instructions | Enhancing LLM Robustness | High | Partly | Use character-level substitutions, insertions, deletions on instruction text only. |
| Context injection | Sentence-level attacks by appending irrelevant or extraneous sentences | PromptRobust | High but adversarial | Partly | Use a milder irrelevant but non-conflicting sentence instead of artificial distractors. |
| Word-level synonym replacement | TextFooler / BertAttack | PromptRobust; Enhancing LLM Robustness | Partial | Not recommended for main experiment | Avoid automatic synonym replacement for math/code because it may change task meaning. |

## 5. Task-by-Perturbation Mapping

This table shows how each project task can borrow perturbation methods from different references.

| Task type | Perturbation to use | Literature source | Method borrowed | Notes |
|---|---|---|---|---|
| Factual QA | Surface noise | POSIX / PromptRobust | spelling errors; character-level typo operations | Apply only to instruction words, not entity names, dates, numbers, or answer-critical content. |
| Factual QA | Formatting changes | POSIX | prompt template changes | Change `Question`, `Answer`, separators, or layout. |
| Factual QA | Context injection | PromptRobust | sentence-level irrelevant sentence insertion | Use mild irrelevant context; do not add new evidence. |
| Factual QA | Paraphrasing | What Did I Do Wrong? / POSIX | rephrase task description or question wording | Must preserve answer target. |
| Math reasoning | Surface noise | POSIX / Enhancing LLM Robustness | spelling errors / DeepWordBug-style instruction edits | Apply only to instruction template, not numbers, units, variables, or formulas. |
| Math reasoning | Formatting changes | POSIX / Haase et al. | prompt template / formatting tweak | Change output format, e.g., `Reasoning:` and `Final answer:`. |
| Math reasoning | Information reordering | Haase et al. | information order | Reorder instruction and answer-format requirements, not the math problem conditions. |
| Code generation | Formatting changes | POSIX / Haase et al. | template and structural changes | Change natural-language layout; keep function signature and examples unchanged. |
| Code generation | Paraphrasing | POSIX / What Did I Do Wrong? | rephrased instruction | Reword instruction such as "write" to "implement"; preserve signature and requirements. |
| Code generation | Surface noise | Enhancing LLM Robustness / PromptRobust | instruction-level character edits | Apply only to natural-language instruction, not code blocks or tests. |
| Open-ended writing | Paraphrasing | POSIX | LLM-generated paraphrases | Strong direct match because POSIX uses Alpaca open-ended prompts. |
| Open-ended writing | Formatting changes | POSIX | template variations | Directly applicable to open-ended prompts. |
| Open-ended writing | Information reordering | Haase et al. | information order | Reorder audience, topic, tone, and length constraints. |
| Open-ended writing | Surface noise | POSIX / Haase et al. | spelling errors / random errors | Can be applied more broadly than in math/code, but still avoid changing topic words. |
| Open-ended writing | Context injection | PromptRobust | irrelevant sentence insertion | Appropriate as a stress-test because open-ended writing may drift with extra context. |

## 6. Evaluation-Level Mapping

| Project evaluation component | Closest literature method | Reference paper | Match level | Project use |
|---|---|---|---|---|
| Output semantic similarity | Sentence embeddings with cosine similarity | Sentence-BERT | Direct | Use SBERT cosine similarity for output-output semantic similarity. |
| Prompt sensitivity | Sensitivity across prompt variations | What Did I Do Wrong? | Conceptual match | Use as support for measuring robustness separately from accuracy. |
| Prompt sensitivity by task/variation type | POSIX sensitivity across prompt variation types and task formats | POSIX | Conceptual match | Compare task-dependent perturbation patterns, but not exact numeric POSIX values. |
| Performance drop | Performance Drop Rate / Average PDR | PromptRobust; Enhancing LLM Robustness | Direct for objective tasks | Use for RQ2 correctness changes in factual QA, math, and code. |
| Semantic preservation of perturbations | Human semantic preservation check / semantic coherence | PromptRobust; POSIX | High | Manually verify perturbations preserve task intent. |
| Repeated sampling / within-model variance | Within-LLM variance from repeated generations | Haase et al. | Direct conceptual support | Estimate sampling-noise baseline before interpreting perturbation effects. |

## 7. What Can Be Claimed As Directly Borrowed

The following methods can be described as directly borrowed or closely followed:

| Method | Source | Safe claim |
|---|---|---|
| Spelling-error operations: insertion, omission, transposition, substitution | POSIX | The surface-noise perturbation follows the spelling-error operations used in POSIX, with task-specific restrictions. |
| Prompt template / formatting changes | POSIX | The formatting perturbation follows the prompt-template variation logic used in POSIX. |
| Information reordering | Haase et al. | The information-order perturbation follows Haase et al.'s information-order prompt variation. |
| Repeated generation for within-model variability | Haase et al. | The sampling-noise baseline is motivated by Haase et al.'s repeated-sampling approach. |
| SBERT cosine similarity | Sentence-BERT | The semantic similarity metric follows the SBERT embedding and cosine-similarity framework. |
| PDR logic for correctness | PromptRobust / Enhancing LLM Robustness | RQ2 can adapt PDR by defining performance according to each task's correctness criterion. |

## 8. What Should Be Described As Adapted

The following methods should not be described as exact replications:

| Project method | Why it is adapted |
|---|---|
| Context injection | PromptRobust uses more adversarial sentence-level attacks; this project uses milder irrelevant context to preserve naturalness. |
| Code generation task | None of the current references directly uses HumanEval or code generation as a main task. |
| Noise-corrected perturbation drift | This is the project's synthesis of prompt perturbation evaluation and repeated-sampling noise correction. |
| Manual perturbation construction | Some papers use algorithms or LLM-generated perturbations; the current pilot uses controlled manual construction. |
| Sentence-BERT drift as the main RQ1 metric | SBERT supports semantic similarity, but the exact noise-corrected drift formula is project-specific. |

## 9. Recommended Paper Framing

Use this structure in the methodology:

```text
First, task types were selected by matching the project goals to task categories used in prior prompt-robustness studies. Factual QA and math reasoning follow the QA and math task categories used in PromptRobust, while open-ended writing follows POSIX's use of Alpaca for open-ended generation. Code generation is added as an extension because it allows objective functional evaluation.

Second, perturbation categories were selected by mapping each planned perturbation to a corresponding method in the literature. Paraphrasing and formatting changes are supported by POSIX and What Did I Do Wrong, information reordering is supported by Haase et al., surface noise is supported by POSIX, PromptRobust, and Enhancing LLM Robustness, and context injection is adapted from PromptRobust's sentence-level perturbations.

Third, evaluation combines prior metrics with the project's noise-correction goal. Sentence-BERT cosine similarity is used for output-level semantic drift, while correctness-based evaluation for RQ2 follows the performance-drop logic used in PromptRobust and Enhancing LLM Robustness. Repeated generation is included following Haase et al.'s warning that single-sample evaluation may confound prompt effects with within-model stochasticity.
```

## 10. Short Explanation For Professor

```text
I am not trying to make the whole project match one paper exactly. Instead, I map each design component to the closest method in the literature. For example, factual QA and math reasoning are aligned with PromptRobust task categories, open-ended writing is aligned with POSIX's Alpaca setting, and code generation is an extension. For perturbations, surface noise can follow POSIX's spelling-error operations, formatting can follow POSIX's template changes, information reordering can follow Haase et al., and context injection can be adapted from PromptRobust's sentence-level attacks. This lets the methodology stay literature-grounded while still allowing the project to combine methods in a new noise-corrected framework.
```
