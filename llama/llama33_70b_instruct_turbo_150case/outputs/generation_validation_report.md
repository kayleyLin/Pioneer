# 150case generation merge validation report

## Inputs

Original shard outputs:

- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\original_generations_shard1.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\original_generations_shard2.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\original_generations_shard3.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\original_generations_shard4.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\original_generations_shard5.csv`

Perturbed shard outputs:

- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\perturbed_generations_shard1.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\perturbed_generations_shard2.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\perturbed_generations_shard3.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\perturbed_generations_shard4.csv`
- `D:\pioneer_kayley_llm\Pioneer\llama\llama33_70b_instruct_turbo_150case\outputs\perturbed_generations_shard5.csv`

## Validation

- Tasks: code_generation, factual_qa, math_reasoning
- Expected items per task: 50
- Expected samples per prompt: 5
- Original rows: 750
- Perturbed rows: 3750
- Duplicate keys: none
- Empty outputs: none
- Original and perturbed item sets: match
