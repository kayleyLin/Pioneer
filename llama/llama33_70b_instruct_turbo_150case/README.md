# Llama 3.3 70B Instruct Turbo 150-case run

本目录是独立的 Llama 3.3 70B 分支，和旧的
`Pioneer/llama/outputs` 中 `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
n50 数据分开保存。

## Model and provider

- Provider: Together AI serverless chat completions
- Endpoint: `https://api.together.ai/v1/chat/completions`
- Model: `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- Temperature: `0.7`
- Top-p: `0.9`
- Repeats: 5 repeated generations per prompt

说明：这个模型不是旧 n50 使用的 8B Lite 模型，因此不能直接作为旧
`Meta-Llama-3-8B-Instruct-Lite` 数据的延续。它应作为新的 70B 模型分支单独分析。

## Scope

本次运行包含三类任务，每类 50 个 case，不包含 `open_ended_writing`：

- `factual_qa`
- `math_reasoning`
- `code_generation`

数据规模：

- Original: `150 cases x 5 repeats = 750 rows`
- Perturbed: `150 cases x 5 perturbations x 5 repeats = 3750 rows`

5 类 perturbation：

- `paraphrasing`
- `reordering`
- `formatting_changes`
- `context_injection`
- `surface_noise`

## Final merged outputs

- `outputs/rq1_llama33_70b_original_generations_150case_three_task.csv`
- `outputs/rq1_llama33_70b_perturbed_generations_150case_three_task.csv`
- `outputs/generation_validation_report.md`

校验结果：

- Original rows: 750
- Perturbed rows: 3750
- Duplicate keys: none
- Empty outputs: none
- Original and perturbed item sets: match
- `model_name`: `meta-llama/Llama-3.3-70B-Instruct-Turbo`

## Shards and logs

5 路并行 shard 输出保存在 `outputs/`：

- `original_generations_shard1.csv` 到 `original_generations_shard5.csv`
- `perturbed_generations_shard1.csv` 到 `perturbed_generations_shard5.csv`

运行日志保存在 `logs/`。本次运行所有 `*.err.log` 均为空。

`outputs/smoke_original_one.csv` 是批量运行前的 1 条 smoke test，不参与最终合并文件。

## Four-task n50 alignment with GPT/main

2026-07-13 已补齐 `open_ended_writing` 的 50 个 case，使本目录也具备和
`Pioneer/outputs` n50 主线一致的四任务结构。

新增 open-ended 输出：

- `outputs/original_generations_n50_open_ended_writing.csv`
- `outputs/perturbed_generations_n50_open_ended_writing.csv`

四任务合并后的推荐文件：

- `outputs/rq1_llama33_70b_original_generations_n50_four_task.csv`
- `outputs/rq1_llama33_70b_perturbed_generations_n50_four_task.csv`
- `outputs/generation_validation_report_n50_four_task.md`

四任务合并后数据量：

- Original: `4 tasks x 50 cases x 5 repeats = 1000 rows`
- Perturbed: `4 tasks x 50 cases x 5 perturbations x 5 repeats = 5000 rows`

每个 task 的数据量与 GPT/main n50 对齐：

- Original: `50 cases x 5 repeats = 250 rows / task`
- Perturbed: `50 cases x 5 perturbations x 5 repeats = 1250 rows / task`

最终校验结果：

- duplicate keys: none
- empty outputs: none
- prompt mismatch: none
- factual QA paraphrasing malformed `Context:`/`Question:` prompts: 0
- math paraphrasing template artifacts: 0
- math paraphrasing ASY diagram deletion cases: 0

说明：原来的 `*_150case_three_task.csv` 文件保留，用于追溯最初三任务运行；如果目标是
和 `Pioneer/outputs` 的 n50 四任务结构一致，应优先使用 `*_n50_four_task.csv`。

### n50 four-task structure check against GPT/main

2026-07-13 已按 `Pioneer/outputs` 中 GPT/main n50 formal generation 的数据结构复查
本目录 Llama 70B n50 四任务文件。推荐使用：

- `outputs/rq1_llama33_70b_original_generations_n50_four_task.csv`
- `outputs/rq1_llama33_70b_perturbed_generations_n50_four_task.csv`

结构与 GPT/main n50 主线一致，主要差异只是 `model_name`：

- GPT/main: `gpt-4o-mini`
- Llama: `meta-llama/Llama-3.3-70B-Instruct-Turbo`

Original 字段：

```text
item_id, task_type, dataset_name, source_index, sample_id,
model_name, temperature, top_p, max_output_tokens,
prompt_text, output_text
```

Perturbed 字段：

```text
item_id, task_type, dataset_name, source_index, perturbation_type,
sample_id, model_name, temperature, top_p, max_output_tokens,
original_prompt, perturbed_prompt, output_text
```

四任务 n50 规模：

| type | rows | structure |
|---|---:|---|
| original | 1000 | `4 tasks x 50 cases x 5 repeats` |
| perturbed | 5000 | `4 tasks x 50 cases x 5 perturbations x 5 repeats` |

每个 task 的规模：

- Original: `50 cases x 5 repeats = 250 rows / task`
- Perturbed: `50 cases x 5 perturbations x 5 repeats = 1250 rows / task`

四类任务：

- `factual_qa`
- `math_reasoning`
- `code_generation`
- `open_ended_writing`

5 类 perturbation：

- `paraphrasing`
- `reordering`
- `formatting_changes`
- `context_injection`
- `surface_noise`

与 GPT/main 的 prompt 对齐检查结果：

- Llama 和 GPT/main 的 `item_id` 集合一致。
- Llama 和 GPT/main 的 original `prompt_text` 一致。
- Llama 和 GPT/main 的 perturbed `original_prompt` / `perturbed_prompt` 一致。
- Llama original key `(item_id, sample_id)` 无重复。
- Llama perturbed key `(item_id, perturbation_type, sample_id)` 无重复。
- 每条 original prompt 都有 `sample_id=1..5`。
- 每条 perturbed prompt 都有 `sample_id=1..5`。
- 空 `output_text` = 0。
- `temperature=0.7`，`top_p=0.9`。

注意：GPT/main n50 在 `Pioneer/outputs` 中主要按 task 拆分成多个文件；本 Llama 70B
目录推荐使用四任务合并文件，然后按 `task_type` 过滤。原来的
`*_150case_three_task.csv` 只包含三类任务，不含 `open_ended_writing`，用于追溯最初
三任务运行，不应作为和 GPT/main n50 四任务对齐时的主输入。

## add100 three-task generation status

2026-07-13 已按 `D:\pioneer_kayley_llm\data_add_100.md` 的规格开始为本目录增加
add100 三任务数据。该批次只包含：

- `factual_qa`
- `math_reasoning`
- `code_generation`

不包含 `open_ended_writing`。

目标数据量：

- Original: `3 tasks x 100 cases x 5 repeats = 1500 rows`
- Perturbed: `3 tasks x 100 cases x 5 perturbations x 5 repeats = 7500 rows`

当前完成状态：

- Original generation 已完成：`1500/1500`
- Perturbed generation 已完成：`7500/7500`
- Perturbed progress: `100%`
- 当前没有继续运行的 Python generation 进程。
- 当前 add100 resume `*.err.log` 均为 0 字节。

当前 add100 shard 行数：

| shard | original | perturbed |
| --- | ---: | ---: |
| shard1 | 300/300 | 1500/1500 |
| shard2 | 300/300 | 1500/1500 |
| shard3 | 300/300 | 1500/1500 |
| shard4 | 300/300 | 1500/1500 |
| shard5 | 300/300 | 1500/1500 |

已完成的 add100 original shard 文件：

- `outputs/add100/original_generations_shard1.csv`
- `outputs/add100/original_generations_shard2.csv`
- `outputs/add100/original_generations_shard3.csv`
- `outputs/add100/original_generations_shard4.csv`
- `outputs/add100/original_generations_shard5.csv`

已完成的 add100 perturbed shard 文件：

- `outputs/add100/perturbed_generations_shard1.csv`
- `outputs/add100/perturbed_generations_shard2.csv`
- `outputs/add100/perturbed_generations_shard3.csv`
- `outputs/add100/perturbed_generations_shard4.csv`
- `outputs/add100/perturbed_generations_shard5.csv`

最终输出：

- `outputs/add100/rq1_llama33_70b_original_generations_add100_three_task.csv`
- `outputs/add100/rq1_llama33_70b_perturbed_generations_add100_three_task.csv`
- `outputs/add100/generation_validation_report_add100_three_task.md`

校验结果：

- Original rows: 1500
- Perturbed rows: 7500
- Original task counts: `500 rows / task`
- Perturbed task counts: `2500 rows / task`
- Perturbation counts: `1500 rows / perturbation`
- Status: `PASS`

### add100 three-task structure check

2026-07-13 已复查 `outputs/add100` 的最终合并文件、shard 文件和 prompt 文件。
add100 是 Llama 70B 的三任务扩展数据：结构和 n50 generation 数据一致，但每类任务
从 50 cases 扩到 100 cases，且不包含 `open_ended_writing`。

推荐使用：

- `outputs/add100/rq1_llama33_70b_original_generations_add100_three_task.csv`
- `outputs/add100/rq1_llama33_70b_perturbed_generations_add100_three_task.csv`
- `outputs/add100/generation_validation_report_add100_three_task.md`

数据规模：

| type | rows | structure |
|---|---:|---|
| original | 1500 | `3 tasks x 100 cases x 5 repeats` |
| perturbed | 7500 | `3 tasks x 100 cases x 5 perturbations x 5 repeats` |

三类任务：

- `code_generation`
- `factual_qa`
- `math_reasoning`

5 类 perturbation：

- `context_injection`
- `formatting_changes`
- `paraphrasing`
- `reordering`
- `surface_noise`

Original 字段：

```text
item_id, task_type, dataset_name, source_index, sample_id,
model_name, temperature, top_p, max_output_tokens,
prompt_text, output_text
```

Perturbed 字段：

```text
item_id, task_type, dataset_name, source_index, perturbation_type,
sample_id, model_name, temperature, top_p, max_output_tokens,
original_prompt, perturbed_prompt, output_text
```

分布检查：

- Original total rows: 1500
- Original unique items: 300
- Original task counts: `500 rows / task`
- Original sample counts: `300 rows / sample_id`
- Perturbed total rows: 7500
- Perturbed unique items: 300
- Perturbed task counts: `2500 rows / task`
- Perturbed perturbation counts: `1500 rows / perturbation`
- Perturbed task x perturbation counts: `500 rows / cell`
- Perturbed sample counts: `1500 rows / sample_id`

一致性检查结果：

- `prompts/add100` combined prompts 与 `tmp_parallel/add100` prompt shards 一致。
- final original CSV 等于 original shard union。
- final perturbed CSV 等于 perturbed shard union。
- original prompt text mismatch: 0
- perturbed prompt text mismatch: 0
- original key `(item_id, sample_id)` 无重复。
- perturbed key `(item_id, perturbation_type, sample_id)` 无重复。
- 每条 original prompt 都有 `sample_id=1..5`。
- 每条 perturbed prompt 都有 `sample_id=1..5`。
- empty `output_text`: 0
- `model_name`: `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `temperature=0.7`，`top_p=0.9`
- Status: `PASS`
