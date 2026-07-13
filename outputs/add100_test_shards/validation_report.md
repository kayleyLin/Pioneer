# add100 generation merge validation report

## Inputs

Original shard outputs:

- `D:\pioneer_kayley_llm\Pioneer\outputs\add100_test_shards\original_generations_shard1.csv`
- `D:\pioneer_kayley_llm\Pioneer\outputs\add100_test_shards\original_generations_shard2.csv`

Perturbed shard outputs:

- `D:\pioneer_kayley_llm\Pioneer\outputs\add100_test_shards\perturbed_generations_shard1.csv`
- `D:\pioneer_kayley_llm\Pioneer\outputs\add100_test_shards\perturbed_generations_shard2.csv`

## Validation

- Tasks: code_generation, factual_qa, math_reasoning
- Expected items per task: 2
- Expected samples per prompt: 5
- Original rows: 30
- Perturbed rows: 150
- Duplicate keys: none
- Empty outputs: none
- Original and perturbed item sets: match
