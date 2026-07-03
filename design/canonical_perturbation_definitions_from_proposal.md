# Canonical Perturbation Definitions From Original Proposal

## 1. Rule

All RQ1 perturbation definitions should follow the original proposal's Section 4.2, Perturbation Design.

The project uses five perturbation types:

```text
1. paraphrasing
2. reordering
3. formatting_changes
4. context_injection
5. surface_noise
```

These are natural, everyday prompt perturbations rather than systematic adversarial attacks.

## 2. Canonical Definitions

| Perturbation type | Canonical definition from proposal | Operational interpretation |
|---|---|---|
| paraphrasing | Meaning-preserving rewording | Rewrite the prompt using different wording while preserving the original meaning and task intent |
| reordering | Rearranging sentences or the order in which information is presented | Change the order of prompt components without adding, removing, or changing information |
| formatting_changes | Converting text to a list or adding/removing polite phrases | Change presentation format, such as list structure, labels, or polite framing, without changing the task |
| context_injection | Adding background information unrelated to the question | Add background/contextual framing that is unrelated to the answer target and does not change task constraints |
| surface_noise | Simulating unintentional spelling or punctuation errors that occur in real user input, such as missing letters or extra spaces, rather than systematic adversarial attacks | Add minor realistic user-input errors, such as typo, missing letter, punctuation error, or extra spacing |

## 3. Manual Equivalence Check

Every perturbed prompt must be manually checked to confirm that it is semantically equivalent to the original prompt.

If a generated or manually written perturbation changes the question itself, the answer target, or the task constraints, it should be rejected and regenerated.

Canonical rule:

```text
Subsequent output differences should reflect a change in wording or presentation, not a change in the question itself.
```

## 4. Implementation Notes

### 4.1 Paraphrasing

Allowed:

```text
Meaning-preserving rewording.
```

Not allowed:

```text
Changing task constraints.
Changing the answer target.
Adding new information.
```

### 4.2 Reordering

Allowed:

```text
Rearranging sentences.
Moving the order of instruction, context, question, examples, or output requirements.
```

Not allowed:

```text
Changing internal logic of a math problem.
Changing code signatures.
Changing evidence or facts.
```

### 4.3 Formatting Changes

Allowed:

```text
Converting prose to a list.
Adding or removing polite phrases.
Changing labels or visual organization.
```

Not allowed:

```text
Changing answer requirements.
Adding new constraints.
Changing the expected output content.
```

### 4.4 Context Injection

Allowed:

```text
Adding background information unrelated to the question.
```

Not allowed:

```text
Adding answer hints.
Adding new evidence.
Adding new conditions.
Changing model role/persona in a way that changes the task.
```

### 4.5 Surface Noise

Allowed:

```text
Missing letters.
Extra spaces.
Minor spelling errors.
Minor punctuation errors.
```

Not allowed:

```text
Systematic adversarial attacks.
Heavy corruption that makes the prompt unreadable.
Changing numbers, formulas, entities, code signatures, or test-relevant examples.
```

