# Perturbation Taxonomy Decision: Follow Original Proposal

## 1. Current Decision

The formal RQ1 perturbation taxonomy follows the original proposal's Section 4.2, Perturbation Design.

The project keeps five perturbation categories:

```text
1. paraphrasing
2. formatting_changes
3. reordering / information_order
4. surface_noise
5. context_injection
```

These definitions should follow the original proposal rather than newly invented replacement categories.

## 2. Definition Of Context Injection

In this project, `context_injection` follows the proposal definition:

```text
Adding background information unrelated to the question.
```

It does **not** mean:

```text
adding new evidence
adding new task conditions
adding hints toward the answer
changing the required output
changing the role/persona of the model
adding systematic adversarial random strings
```

## 3. Literature Basis

| Project category | Closest literature method | Reference |
|---|---|---|
| context_injection | sentence-level attacks that append irrelevant or extraneous sentences | PromptRobust |
| context_injection | CheckList / StressTest-style distractor insertion | PromptRobust |

Important adaptation:

```text
PromptRobust uses sentence-level attacks in a more adversarial way. This project adapts the idea into a milder natural-context perturbation, because the goal is to test prompt sensitivity under plausible prompt variation rather than maximize failure.
```

## 4. Why Keep It

`context_injection` remains part of the taxonomy because it was included in the original proposal and adds a perturbation type that is different from the other four:

| Perturbation type | What changes |
|---|---|
| paraphrasing | wording |
| formatting_changes | layout / structure |
| reordering | information order |
| surface_noise | low-level textual form |
| context_injection | unrelated background information |

This gives the taxonomy broader coverage of prompt variation.

## 5. Risk And Control Rule

The main risk is that added context may subtly change task meaning. To control this, context injection must follow strict rules:

```text
1. The injected context must be irrelevant but not contradictory.
2. It must not provide a hint toward the correct answer.
3. It must not add a new constraint.
4. It must not change the requested output format.
5. It must be short.
6. It must be manually checked for semantic equivalence.
```

## 6. Task-Specific Rules

### Factual QA

Allowed:

```text
For a trivia practice question, answer this:
```

Avoid:

```text
adding background knowledge
adding source/evidence claims
adding topic hints that point to the answer
```

### Math Reasoning

Allowed:

```text
For a basic arithmetic word-problem exercise, solve this:
```

Avoid:

```text
adding new arithmetic hints
adding reminders that change strategy
adding comments about which operation to use
```

### Code Generation

Allowed:

```text
For a beginner Python programming exercise, write...
```

Avoid:

```text
adding library constraints
adding performance requirements
adding style requirements
changing whether explanation is allowed
```

### Open-Ended Writing

Allowed:

```text
For a school wellness discussion, describe...
```

Avoid:

```text
changing audience too strongly
changing genre
changing required length
adding a new argument direction
```

## 7. Recommended Paper Wording

```text
The study follows the original proposal and uses five prompt perturbation categories: paraphrasing, reordering, formatting changes, context injection, and surface noise. Paraphrasing is defined as meaning-preserving rewording; reordering rearranges sentences or the order in which information is presented; formatting changes include converting text to a list or adding/removing polite phrases; context injection adds background information unrelated to the question; and surface noise simulates unintentional spelling or punctuation errors found in real user input, such as missing letters or extra spaces. Each perturbed prompt is manually checked to ensure that it remains semantically equivalent to the original prompt.
```

## 8. Practical Implication For Data Collection

The existing RQ1b pilot prompt file already follows this five-category taxonomy:

```text
prompts/rq1b_pilot_perturbed_prompts.csv
```

The `context_injection` rows in that file should remain part of the formal perturbation pilot unless a later review finds that a specific injected phrase changes the task meaning.
