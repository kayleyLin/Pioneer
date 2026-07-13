# Llama 3.3 70B n50 four-task validation report

## Validation

- Tasks: code_generation, factual_qa, math_reasoning, open_ended_writing
- Expected items per task: 50
- Expected samples per prompt: 5
- Original rows: 1000
- Perturbed rows: 5000
- Duplicate keys: none
- Empty outputs: none
- Prompt mismatch: none
- Original task counts: {'code_generation': 250, 'factual_qa': 250, 'math_reasoning': 250, 'open_ended_writing': 250}
- Perturbed task counts: {'code_generation': 1250, 'factual_qa': 1250, 'math_reasoning': 1250, 'open_ended_writing': 1250}
- Perturbation counts: {'context_injection': 1000, 'formatting_changes': 1000, 'paraphrasing': 1000, 'reordering': 1000, 'surface_noise': 1000}

## Status

PASS
