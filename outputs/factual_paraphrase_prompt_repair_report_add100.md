# Factual QA Paraphrasing Prompt Repair Report

- Repaired rows: 100
- Rows with rewrite-prefix residue before repair: 17
- Rows with rewrite-prefix residue after repair: 0

## Before Parse Status

| status                                   |   rows |
|:-----------------------------------------|-------:|
| question_only_no_marker                  |     89 |
| parsed_full_context_question_with_prefix |     11 |

## After Parse Status

| status                       |   rows |
|:-----------------------------|-------:|
| parsed_full_context_question |    100 |

## Conclusion

PASS. The fixed factual-QA paraphrasing prompts all contain a full `Context:` block and a clean `Question:` block.
