# Correctness Evaluator Notes

## General Rule

The evaluator is intentionally conservative and fully automatic. It should label clearly correct outputs as correct. If a row cannot be matched or compared reliably by the automatic rule, it is labeled incorrect rather than sent to manual review.

Rationale:

```text
The project does not use manual correctness judging because the review workload would be too large and difficult to reproduce consistently.
```

## Factual QA

Recommended method:

```text
containment first + token F1 backup
```

Normalization:

```text
lowercase
remove punctuation
remove English articles: a, an, the
collapse whitespace
```

Automatic correct:

```text
the normalized reference answer appears in the normalized output
```

Performance score:

```text
if normalized reference answer appears in normalized output:
    factual_score = 1.0
else:
    factual_score = SQuAD-style token F1 between reference answer and full output
```

Binary correctness:

```text
is_correct is left blank for factual_qa
```

For RQ2 performance-drop analysis, `factual_score` / `performance_score` is used directly. No threshold is applied.

Implemented output fields:

```text
factual_containment_match
factual_token_f1
performance_score
```

## Math Reasoning

Method:

```text
final-answer extraction followed by normalized and symbolic comparison
```

Extraction priority:

```text
boxed answer
phrases such as "final answer is" or "answer is"
last numeric or simple mathematical expression in the output
```

Automatic correct:

```text
normalized answer strings match
or SymPy confirms symbolic/numeric equivalence
```

Automatic incorrect:

```text
answer extraction fails
or the parser cannot safely compare the extracted answer with the reference
```

## Code Generation

Method:

```text
extract Python code and execute HumanEvalPack tests in a temporary subprocess
```

Automatic correct:

```text
the extracted function passes all tests for the corresponding HumanEvalPack item
```

Automatic incorrect:

```text
the code fails tests, raises an exception, has syntax errors, or times out
```

Automatic unresolved metadata cases:

```text
the HumanEvalPack test metadata cannot be loaded
```

These cases indicate a pipeline/data problem. In ordinary generated-code failures, syntax errors, runtime errors, assertion failures, and timeouts are labeled incorrect.

## Important Limitation

The automatic evaluator is part of the measurement pipeline, not a perfect truth source. The resulting correctness rates should be interpreted as automatic benchmark-style correctness under conservative matching rules.
