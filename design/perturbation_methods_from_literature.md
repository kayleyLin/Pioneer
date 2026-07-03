# Perturbation Methods Found In The Reference Literature

## 1. Purpose

This document lists the prompt perturbation methods that appear in the project reference papers and maps them to the current Pioneer perturbation taxonomy.

Current project perturbation types:

```text
P1 paraphrasing
P2 formatting changes
P3 information reordering
P4 surface noise
P5 context injection
```

Main conclusion:

```text
No single reference paper uses exactly the full five-category taxonomy in the same way as this project.
However, the five categories can be strongly supported by combining methods from Haase et al., POSIX, PromptRobust, Enhancing LLM Robustness, and What Did I Do Wrong.
```

## 2. Direct Method Mapping Table

| Project perturbation type | Same or closest method in literature | Reference paper | How the perturbation is added in the paper | Match level | How to use in this project |
|---|---|---|---|---|---|
| Paraphrasing | Paraphrases / rewording | POSIX | Uses GPT-3.5-Turbo to generate 20 paraphrases for each original prompt while preserving original intent and meaning | Very high | Use as the main literature support for paraphrasing. If using LLM-generated paraphrases later, manually verify meaning preservation. |
| Paraphrasing | Prompt rephrasings | What Did I Do Wrong? | Uses LLMs to rephrase the task description by changing length or adding unnecessary words while keeping meaning the same | Very high | Supports using natural rephrasing as a sensitivity test. Especially useful for explaining prompt sensitivity without ground-truth dependence. |
| Paraphrasing | Phrasing change | Haase et al. | Uses minor phrasing / surface edits to the baseline prompt to test sensitivity to semantic rewordings | Very high | Strong support for a lightweight, non-adversarial paraphrasing condition. |
| Formatting changes | Prompt templates | POSIX | Designs 20 prompt templates that keep core meaning but alter prompt structure, labels, separators, capitalization, and answer markers | Very high | Use as the strongest support for formatting/template perturbation. |
| Formatting changes | Formatting tweak | Haase et al. | Uses a structural constraint / formatting tweak, such as imposing strict output constraints | Very high | Supports treating formatting as its own perturbation type rather than merging it with paraphrase. |
| Information reordering | Information order | Haase et al. | Reorders elements while keeping the same information as the baseline prompt | Direct match | This is the clearest direct support for the project's reordering category. |
| Surface noise | Spelling errors | POSIX | Randomly selects 1, 2, 4, or 8 tokens and applies insertion, omission, transposition, or substitution errors | Very high | Use as the most precise rule-based model for project surface noise. For math/code, apply only to instruction words, not task-critical content. |
| Surface noise | Character-level attacks | PromptRobust | Uses TextBugger and DeepWordBug to add, delete, repeat, replace, or permute characters in words | High | Supports typo/noise perturbations, but project should weaken these from adversarial attacks into natural surface noise. |
| Surface noise | DeepWordBug | Enhancing LLM Robustness | Uses character-level substitutions, insertions, and deletions on task-specific instructions | High | Strong support for instruction-level surface noise. |
| Surface noise | Random errors / typo robustness | Haase et al. | Injects typographical and syntactic noise to assess model error tolerance | Very high | Supports using typo robustness as a natural prompt perturbation. |
| Context injection | Sentence-level attacks | PromptRobust | Uses StressTest and CheckList to append irrelevant or extraneous sentences to the end of prompts to distract LLMs | High, but more adversarial | Closest support for context injection. Project should use irrelevant but non-conflicting sentences, not aggressive adversarial distractors. |
| Context injection | StressTest | PromptRobust | Appends repeated logic-like distractor sentences such as "and true is true" or "and false is not true" | Partial | Use only as background support; the exact string is too artificial for this project. |
| Context injection | CheckList | PromptRobust | Appends random alphabet/digit sequences to the end of prompts | Partial | Not recommended for main project because it is closer to adversarial noise than natural context injection. |
| Word-level replacement | TextFooler / BertAttack | PromptRobust | Replaces words with synonyms or contextually similar words | Partial | Not recommended as a main automatic method for math/code because synonym replacement may change task meaning. |
| Word-level replacement | TextFooler | Enhancing LLM Robustness | Uses word replacements by counter-fitted GloVe embeddings | Partial | Useful as literature background, but not ideal for current natural perturbation taxonomy. |
| Semantic-level variation | Semantic-level attacks | PromptRobust | Simulates linguistic behavior from people using different languages/countries | Low for current design | Not recommended for current pilot; too different from the five current perturbation categories. |

## 3. Best Literature Source For Each Project Perturbation

| Project perturbation | Best reference to cite first | Secondary references |
|---|---|---|
| Paraphrasing | POSIX | What Did I Do Wrong?; Haase et al. |
| Formatting changes | POSIX | Haase et al.; What Did I Do Wrong? |
| Information reordering | Haase et al. | POSIX, as broader template-structure support |
| Surface noise | POSIX | PromptRobust; Enhancing LLM Robustness; Haase et al. |
| Context injection | PromptRobust | CheckList / StressTest tradition; use cautiously |

## 4. Exact Methods That Can Be Adapted Directly

### 4.1 POSIX Spelling Errors

POSIX gives the clearest rule-based method for spelling errors:

```text
1. Randomly select 1, 2, 4, or 8 tokens.
2. Apply one of four spelling error types:
   a. insertion: add a random letter within the token
   b. omission: delete a letter
   c. transposition: swap two adjacent letters
   d. substitution: replace a letter with an adjacent keyboard letter
3. Use different random seeds to generate multiple variants.
```

Recommended project adaptation:

```text
Use the same four error operations, but apply them only to instruction text.
Do not apply them to entity names, numbers, formulas, code signatures, examples, or answer-critical content.
```

### 4.2 POSIX Prompt Templates

POSIX changes prompt templates while preserving meaning. Examples include changing:

```text
Q: ... A:
Question: ... Answer:
QUESTION ... ANSWER
Question: ... || Answer:
Question - ... Answer -
```

Recommended project adaptation:

```text
Use template/format changes such as labels, separators, bullet points, numbered fields, or answer markers.
Keep the actual task content unchanged.
```

### 4.3 POSIX Paraphrases

POSIX uses GPT-3.5-Turbo to generate paraphrases while preserving intent.

Recommended project adaptation:

```text
For the current pilot, manual paraphrasing is acceptable.
For the formal version, LLM-generated paraphrases can be used if every paraphrase is manually checked for semantic equivalence.
```

### 4.4 Haase Information Order

Haase et al. include an "Information Order" prompt variation:

```text
Reorder elements but keep the same information as in the baseline prompt.
```

Recommended project adaptation:

```text
Move instruction, constraints, output requirements, examples, or context fields into a different order.
Do not reorder internal math conditions, code signatures, or factual evidence in a way that changes the task.
```

### 4.5 PromptRobust Sentence-Level Attacks

PromptRobust appends irrelevant or extraneous sentences to the end of prompts.

Recommended project adaptation:

```text
Use a mild context-injection version:
Add one irrelevant but non-conflicting sentence.
Avoid random strings or repeated artificial distractors unless the goal is an adversarial stress test.
```

## 5. Methods That Are Similar But Should Not Be Used Directly

| Literature method | Why not use directly as-is | Safer project adaptation |
|---|---|---|
| TextFooler / BertAttack word replacement | May change facts, math conditions, or code requirements | Use manual paraphrase instead |
| DeepWordBug on arbitrary prompt tokens | May corrupt entities, numbers, code signatures, or examples | Apply surface noise only to non-critical instruction words |
| CheckList random alphanumeric strings | Too artificial; closer to adversarial distraction than natural context | Use a natural irrelevant sentence instead |
| StressTest repeated logic strings | Artificial and may not reflect normal user prompt variation | Use one mild distractor sentence |
| PromptRobust semantic-level cross-lingual style | Not aligned with current five perturbation types | Leave as possible future extension |

## 6. Recommended Perturbation Rules For This Project

Based on the literature, the safest project rules are:

```text
1. Apply perturbations primarily to the instruction part of the prompt.
2. Keep task-critical content unchanged.
3. Use one perturbation type at a time.
4. For surface noise, use POSIX-style insertion, deletion, transposition, and substitution.
5. For formatting, use POSIX-style template changes and Haase-style structural constraints.
6. For reordering, use Haase-style information order changes.
7. For context injection, use a softened version of PromptRobust sentence-level attacks.
8. Manually verify that the perturbed prompt preserves the original task intent.
```

## 7. Table For Paper Methodology

This table can be copied into the methodology section.

| Perturbation category | Operational definition in this study | Literature basis |
|---|---|---|
| Paraphrasing | Reword the instruction while preserving the original task intent | POSIX paraphrases; What Did I Do Wrong prompt rephrasings; Haase phrasing change |
| Formatting changes | Change labels, separators, bullets, answer markers, or structural layout without changing content | POSIX prompt templates; Haase formatting tweak |
| Information reordering | Reorder prompt elements while keeping the same information | Haase information order prompt |
| Surface noise | Add minor spelling, punctuation, capitalization, or typographical errors to non-critical instruction words | POSIX spelling errors; PromptRobust character-level attacks; Enhancing LLM Robustness DeepWordBug; Haase random errors |
| Context injection | Add an irrelevant but non-conflicting sentence to the prompt | PromptRobust sentence-level attacks using StressTest and CheckList, adapted into a milder natural-context version |

## 8. Key Caveat

The project should not claim that all five perturbation types are copied exactly from one paper.

Better wording:

```text
The five perturbation categories are synthesized from prior prompt-robustness studies. Four categories - paraphrasing, formatting changes, information reordering, and surface noise - have close or direct precedent in Haase et al. and POSIX. The fifth category, context injection, is adapted from PromptRobust's sentence-level perturbations, but softened to preserve naturalness and task intent.
```

