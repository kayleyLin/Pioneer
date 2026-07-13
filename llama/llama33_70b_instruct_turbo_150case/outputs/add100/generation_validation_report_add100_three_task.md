# Llama 3.3 70B add100 three-task validation report

## Validation

- Tasks: code_generation, factual_qa, math_reasoning
- Expected items per task: 100
- Expected samples per prompt: 5
- Original rows: 1500
- Perturbed rows: 7500
- Original task counts: {'code_generation': 500, 'factual_qa': 500, 'math_reasoning': 500}
- Perturbed task counts: {'code_generation': 2500, 'factual_qa': 2500, 'math_reasoning': 2500}
- Perturbation counts: {'context_injection': 1500, 'formatting_changes': 1500, 'paraphrasing': 1500, 'reordering': 1500, 'surface_noise': 1500}

## Status

PASS
