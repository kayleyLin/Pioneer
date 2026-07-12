# Factual QA Paraphrasing Prompt Repair Report

- Repaired rows: 50
- Rows with rewrite-prefix residue before repair: 7
- Rows with rewrite-prefix residue after repair: 0

## Before Parse Status

| status                                   |   rows |
|:-----------------------------------------|-------:|
| question_only_no_marker                  |     43 |
| parsed_full_context_question_with_prefix |      7 |

## After Parse Status

| status                       |   rows |
|:-----------------------------|-------:|
| parsed_full_context_question |     50 |

## Conclusion

PASS. The fixed factual-QA paraphrasing prompts all contain a full `Context:` block and a clean `Question:` block.
