# Qwen add100 three-task generation plan

Last updated: 2026-07-14

Execution status: **COMPLETE (2026-07-14)**

- Smoke test passed with `qwen-plus`; model and sampling configuration were not changed.
- Add100 generation completed: 1,500 original rows and 7,500 perturbed rows.
- `qwen/outputs/add100_generation_validation_report.md`: **PASS**.
- Three-task n=150 merge completed: 2,250 original rows, 11,250 perturbed rows, and 450 prompts.
- `qwen/outputs/n150_three_task_merge_validation_report.md`: **PASS**.
- Parameterized n=150 SBERT outputs were generated under `qwen/outputs/`.
- Parameterized n=150 correctness outputs were generated under `qwen/rq2_outputs_n150/`.
- Independent final audit found zero duplicate generation/analysis keys, zero empty generation outputs, complete SBERT/correctness key coverage, and only the expected undefined correctness-retention values where original performance is zero.
- RFRI/BCRI zero-tuning external validation was intentionally not run, per scope.

## 1. Objective

为 Qwen 分支增加与 GPT/main `add100` 完全相同的一批 cases，使后续 GPT、Qwen、Llama 可以按相同 `item_id` 做配对比较。

本项目中的 `add100` 不是总共增加 100 个 item，而是：

- `factual_qa`：新增 100 items；
- `math_reasoning`：新增 100 items；
- `code_generation`：新增 100 items；
- 不增加 `open_ended_writing`。

每个 original prompt 和每个 perturbed prompt 均生成 5 次 sampled outputs。模型和采样配置应与现有 Qwen n50 保持一致：

```text
provider: DashScope OpenAI-compatible API
endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
model: qwen-plus
temperature: 0.7
top_p: 0.9
n_samples_per_prompt: 5
system prompt: You are a helpful assistant. Answer the user's prompt directly.
```

本计划只规定生成、恢复、合并和验证流程。执行前不重新抽样，不修改 n50 文件，也不调用 GPT/OpenAI API。

## 2. Current-state findings

### 2.1 Existing Qwen n50

Qwen 当前已有四任务 n50 generation。三项客观 correctness 任务的主线数据为：

Original：

- `qwen/outputs/rq1_qwen_original_generations_n50_factual_qa.csv`
- `qwen/outputs/rq1_qwen_original_generations_n50_math_reasoning.csv`
- `qwen/outputs/rq1_qwen_original_generations_n50_code_generation.csv`

Perturbed：

- factual 使用 `qwen/outputs/rq1_qwen_perturbed_generations_n50_factual_qa_fixed.csv`
- math 使用 `qwen/outputs/rq1_qwen_perturbed_generations_n50_math_reasoning_fixed.csv`
- code 使用 `qwen/outputs/rq1_qwen_perturbed_generations_n50_code_generation.csv`

已检查这些文件：每个 task 的 original 为 250 行，perturbed 为 1,250 行；模型均为 `qwen-plus`，无空输出和重复 generation key。

### 2.2 Add100 prompt corpus

应复用当前共享 prompt files，而不是为 Qwen 重新抽样：

Original：

- `prompts/rq1_sampled_original_prompts_add100_three_task.csv`
- 300 rows = 100 items/task × 3 tasks。

Perturbed：

- `prompts/rq1_formal_perturbed_prompts_add100_factual_qa.csv`
- `prompts/rq1_formal_perturbed_prompts_add100_math_reasoning.csv`
- `prompts/rq1_formal_perturbed_prompts_add100_code_generation.csv`
- 每个文件 500 rows；合计 1,500 rows。

本次只读 preflight 已确认：

- add100 original item IDs 与三任务 n50 item IDs 的交集为 0；
- 300 个 original `item_id` 唯一；
- 1,500 个 `(item_id, perturbation_type)` 唯一；
- 每个 item 都有五种 perturbations；
- original 与 perturbed item sets 完全一致；
- factual paraphrasing malformed shape = 0；
- factual rewrite/paraphrase residue = 0；
- math template artifacts = 0；
- math `[asy]` diagram deletion = 0；
- GPT add100 generation 中保存的 prompts 与当前 prompt corpus mismatch = 0。

因此当前 prompts 已包含 2026-07-13 的 factual/math paraphrasing 修复。Qwen 必须直接从这些文件生成，不能使用旧 shard、旧 prompt copy 或修复前 generation。

### 2.3 Expected generation volume

| condition | prompt rows | repeats | expected generation rows/API calls |
|---|---:|---:|---:|
| Original | 300 | 5 | 1,500 |
| Perturbed | 1,500 | 5 | 7,500 |
| Total | 1,800 | 5 | 9,000 |

Perturbed 的 7,500 行应满足：

- 2,500 rows/task；
- 1,500 rows/perturbation type；
- 25 rows/item，即 5 perturbations × 5 repeats。

## 3. Output layout

Add100 应与 n50 分开保存，直到全部验证通过。

```text
qwen/
├── tmp_parallel/
│   └── add100/
│       ├── manifest.csv
│       ├── original_prompts_shard1.csv ... shard5.csv
│       └── perturbed_prompts_shard1.csv ... shard5.csv
├── add100_logs/
│   ├── original_shard1.log ... shard5.log
│   └── perturbed_shard1.log ... shard5.log
└── outputs/
    ├── add100_shards/
    │   ├── original_generations_shard1.csv ... shard5.csv
    │   └── perturbed_generations_shard1.csv ... shard5.csv
    ├── rq1_qwen_original_generations_add100_three_task.csv
    ├── rq1_qwen_perturbed_generations_add100_three_task.csv
    └── add100_generation_validation_report.md
```

Optional n150 合并文件：

```text
qwen/outputs/rq1_qwen_original_generations_n150_three_task.csv
qwen/outputs/rq1_qwen_perturbed_generations_n150_three_task_fixed.csv
qwen/outputs/n150_three_task_merge_validation_report.md
```

不要把 add100 rows 直接追加到现有 n50 per-task files。保留独立 add100 文件可以审计、恢复和重新合并。

## 4. Implementation phases

## Phase 0 — Freeze configuration and protect existing data

1. 记录现有 Qwen n50 文件的 row counts 和 SHA256。
2. 记录以下固定输入的 SHA256：
   - 300-row original prompt file；
   - 三个 500-row perturbed prompt files；
   - `config/rq1_generation_config.json`；
   - 两个 Qwen generation scripts。
3. 确认 `qwen_api.txt` 存在，但不得把 key 写入日志或 validation report。
4. 锁定模型为现有 n50 使用的 `qwen-plus`。如果 smoke test 返回模型不可用，不要静默换成其他 Qwen model；先停止并记录，再决定是否接受 model-version change。
5. 正式运行前保存 run manifest，包括开始时间、model、endpoint、temperature、top_p、system prompt、input hashes 和 script hashes。

建议新增：

```text
qwen/add100_run_manifest.json
```

## Phase 1 — Build Qwen-specific prompt shards

可直接复用 `src/61_split_add100_generation_shards.py`，但输出必须写入 `qwen/tmp_parallel/add100/`，不要复用 GPT 或 Llama 的临时 shards。

```powershell
python src/61_split_add100_generation_shards.py `
  --original-inputs prompts/rq1_sampled_original_prompts_add100_three_task.csv `
  --perturbed-inputs `
    prompts/rq1_formal_perturbed_prompts_add100_factual_qa.csv `
    prompts/rq1_formal_perturbed_prompts_add100_math_reasoning.csv `
    prompts/rq1_formal_perturbed_prompts_add100_code_generation.csv `
  --output-dir qwen/tmp_parallel/add100 `
  --shards 5 `
  --expected-items-per-task 100
```

Expected shard manifest：

| shard | original prompt rows | expected original outputs | perturbed prompt rows | expected perturbed outputs |
|---:|---:|---:|---:|---:|
| 1–5 | 60 each | 300 each | 300 each | 1,500 each |

`src/61` 已包含关键 prompt validation：task set、五种 perturbations、duplicate keys、factual paraphrase shape、math template artifacts 和 `[asy]` preservation。任何一项失败时不得开始 API generation。

## Phase 2 — Add a Qwen add100 runner

建议新增：

```text
src/78_run_qwen_add100_parallel.py
```

该 runner 应复用而不是复制以下 generation logic：

- original：`src/07_generate_rq1_outputs_qwen.py`
- perturbed：`src/11_generate_rq1b_perturbed_outputs_qwen.py`

Runner requirements：

1. 支持 `--phase smoke|original|perturbed|all`。
2. 读取 `qwen/tmp_parallel/add100/manifest.csv`，计算每个 shard 的 expected rows。
3. 五个 shard 使用独立 output CSV，避免并发写同一文件。
4. 初始并发建议为 5 个 shard processes × 每个 process `workers=1`，即最多 5 个并行 API requests。
5. 如果连续出现 429，应降低为 2–3 个 shard processes；不要通过增加线程强行加速。
6. 每个 shard 按 generation key 断点续跑：
   - original key = `(item_id, sample_id)`；
   - perturbed key = `(item_id, perturbation_type, sample_id)`。
7. 重跑时只补 missing keys；不得覆盖已经完成且非空的 rows。
8. HTTP 408、429、5xx、timeout 和 connection reset 使用 exponential backoff + jitter；最终失败必须返回 non-zero exit code。
9. 每次 API response 必须检查 `choices[0].message.content` 非空。
10. Runner 不删除 shards。只有最终 merge 和 validation PASS 后，才可以选择归档 shards。
11. API key 只从 `DASHSCOPE_API_KEY` 环境变量读取，日志只记录 `API_KEY=set/not set`。
12. 每个 shard 完成后立即运行轻量 validation，并输出 `actual/expected`。

现有 generation scripts 已支持 `--input`、`--output` 和 resume keys。正式实现时可由 runner 启动它们，不需要重写 Qwen API request body。

## Phase 3 — Smoke test

先在独立 smoke output 中测试，不要写入正式 shards：

```powershell
$env:DASHSCOPE_API_KEY = (Get-Content qwen_api.txt -Raw).Trim()

python src/07_generate_rq1_outputs_qwen.py `
  --input qwen/tmp_parallel/add100/original_prompts_shard1.csv `
  --output qwen/outputs/add100_smoke_original.csv `
  --samples 1 `
  --limit-prompts 2

python src/11_generate_rq1b_perturbed_outputs_qwen.py `
  --input qwen/tmp_parallel/add100/perturbed_prompts_shard1.csv `
  --output qwen/outputs/add100_smoke_perturbed.csv `
  --samples 1 `
  --limit-rows 2
```

Smoke acceptance：

- API authentication succeeds；
- response model path accepts `qwen-plus`；
- two output files are valid UTF-8 CSV；
- output text is non-empty；
- `model_name=qwen-plus`；
- prompt text in output exactly matches the input row；
- temperature/top_p metadata are 0.7/0.9；
- resume rerun produces zero duplicate keys。

Smoke files不能与 formal shard outputs 合并。

## Phase 4 — Formal generation with resume

Recommended order：

1. 先完成 1,500 original rows；
2. original validation PASS 后再开始 7,500 perturbed rows；
3. 每轮运行后读取五个 shard logs 和 row counts；
4. 若有 shard 未完成，重新执行相同 phase，runner 只补 missing keys。

Planned commands：

```powershell
$env:DASHSCOPE_API_KEY = (Get-Content qwen_api.txt -Raw).Trim()

python src/78_run_qwen_add100_parallel.py --phase original --max-parallel 5
python src/78_run_qwen_add100_parallel.py --phase perturbed --max-parallel 5
```

不要只检查 CSV 的物理行数。完成度必须基于 unique generation keys，因为 interrupted append 或重复恢复可能使物理行数看起来正确但 key set 不完整。

## Phase 5 — Merge and strict validation

建议新增：

```text
src/79_merge_validate_qwen_add100.py
```

可以复用 `src/62_merge_validate_add100_outputs.py` 的核心逻辑，但 Qwen 版本必须增加 prompt equality、model/config 和 repaired-paraphrase checks。

Final add100 acceptance criteria：

### Original

- rows = 1,500；
- unique items = 300；
- task counts = 500 each；
- unique `(item_id, sample_id)` = 1,500；
- every item has sample set `{1,2,3,4,5}`；
- empty outputs = 0；
- `prompt_text` 与 current original prompt file exact match；
- `model_name=qwen-plus` for all rows；
- temperature = 0.7 and top_p = 0.9 for all rows。

### Perturbed

- rows = 7,500；
- unique items = 300；
- task counts = 2,500 each；
- perturbation counts = 1,500 each；
- unique `(item_id, perturbation_type, sample_id)` = 7,500；
- every item-perturbation has sample set `{1,2,3,4,5}`；
- empty outputs = 0；
- `original_prompt` 和 `perturbed_prompt` 与三个 current perturbed prompt files exact match；
- factual paraphrasing malformed/residue rows = 0；
- math template artifacts = 0；
- math `[asy]` deletion = 0；
- `model_name=qwen-plus` for all rows；
- temperature = 0.7 and top_p = 0.9 for all rows。

只有全部检查通过后才写出：

```text
qwen/outputs/rq1_qwen_original_generations_add100_three_task.csv
qwen/outputs/rq1_qwen_perturbed_generations_add100_three_task.csv
qwen/outputs/add100_generation_validation_report.md
```

Validation report 应包含 input/output hashes、row counts、task/perturbation/model counts、duplicate/empty/mismatch counts、sample-set status 和最终 `PASS/FAIL`。

## Phase 6 — Build the three-task n150 corpus

Add100 PASS 后，再创建 n150 combined files。建议新增：

```text
src/80_merge_validate_qwen_n150_three_task.py
```

Merge inputs：

Original：

- Qwen n50 factual/math/code original files；
- Qwen add100 original file。

Perturbed：

- n50 factual `_fixed.csv`；
- n50 math `_fixed.csv`；
- n50 code standard perturbed file；
- Qwen add100 perturbed file。

Expected n150 output：

| condition | formula | rows |
|---|---:|---:|
| Original | 3 tasks × 150 items × 5 repeats | 2,250 |
| Perturbed | 3 tasks × 150 items × 5 perturbations × 5 repeats | 11,250 |

N150 validation：

- 150 unique items/task；
- add100 与 n50 item intersection = 0；
- no duplicate generation keys；
- no empty outputs；
- sample sets complete；
- original/perturbed item sets match；
- factual/math n50 portion comes from fixed files；
- add100 portion matches current fixed prompts；
- all rows use `qwen-plus` and the same sampling metadata。

## Phase 7 — Downstream analysis after generation

Generation 完成不代表现有 n50 analysis scripts 会自动处理 n150。当前以下脚本或路径包含 n50 hardcoding：

- `src/31_qwen_analyze_rq1_n50_baseline_sbert.py`
- `src/32_qwen_analyze_rq1_n50_perturbations_sbert.py`
- `src/39_build_external_model_rq2_fig4_data.py`

后续应将它们参数化为接受：

- prompt metadata path；
- original generation files/path；
- perturbed generation files/path；
- output tag，例如 `n150_three_task`；
- task list，仅 factual/math/code。

不要通过修改 n50 文件或把 n150 文件伪装成 n50 filename 来运行分析。

建议后续输出命名：

```text
qwen/outputs/sbert_rq1_n150_three_task_baseline_by_item.csv
qwen/outputs/sbert_rq1_n150_three_task_perturbation_effects_by_item.csv
qwen/rq2_outputs_n150/...
```

## 5. Recovery strategy

### API interruption or rate limit

- 保留所有已完成 shard CSV 和 logs；
- 检查 CSV 可解析性和 unique key set；
- 重跑相同 phase，只生成 missing keys；
- 429 持续出现时降低并发，不改变 sample IDs；
- 不把 incomplete shards 合并成 final output。

### One bad or empty row

- 把 bad generation key 写入 repair manifest；
- 从该 shard 删除或隔离对应 bad row；
- 仅重跑该 key；
- validation report 记录 repair count 和原因。

### Prompt mismatch

- 不允许在 merged CSV 内直接改 prompt 字符串；
- 查明使用了哪个旧 shard/input；
- 删除受影响 condition 的 generated rows并从 current prompt file 重跑；
- factual/math paraphrasing mismatch 时至少重跑该 task 的全部 paraphrasing condition，避免混用旧/新 prompts。

### Model unavailable

- 不自动切换模型；
- 保存 HTTP status 和非敏感 error body；
- 若必须改模型，生成新 tag 和新文件名，不能与 `qwen-plus` n50 合并为同一个 model corpus。

## 6. Cost and runtime controls

- 正式计划包含 9,000 次 API completions；执行前应先确认 DashScope 账户 quota。
- Monetary cost 取决于实际 input/output token counts 和执行时价格，本计划不使用静态价格估算。
- Smoke test 通过后可先完成 original 1,500 calls，检查输出长度和 token usage，再决定 perturbed phase 的并发。
- 如 API response 提供 usage metadata，runner 应聚合 prompt/completion tokens 到 manifest/report，但不需要在 generation CSV 中增加会破坏现有 schema 的列。
- 每完成一个 shard 就记录 elapsed time、successful calls、retry counts 和 token usage，便于估计剩余任务。

## 7. Definition of done

Qwen add100 只有同时满足以下条件才算完成：

1. 固定复用当前 repaired add100 prompts，没有重新抽样。
2. 生成 1,500 original rows 和 7,500 perturbed rows。
3. 所有 9,000 rows 均为 `qwen-plus`、非空、key 唯一。
4. 每个 condition 的 five-repeat sample set 完整。
5. Generation 中保存的 prompt fields 与 current prompt corpus exact match。
6. Factual/math paraphrasing repair checks 全部为 0 errors。
7. `qwen/outputs/add100_generation_validation_report.md` 状态为 `PASS`。
8. Add100 standalone files、shards、logs 和 manifest 均保留可审计。
9. N50 文件未被覆盖或修改。
10. 如果创建 n150，n150 merge report 也必须为 `PASS`。

## 8. Recommended execution order

```text
1. Freeze hashes/config
2. Build and validate Qwen-specific prompt shards
3. Implement add100 runner and strict validator
4. Run isolated smoke test
5. Generate/resume original shards
6. Validate original completeness
7. Generate/resume perturbed shards
8. Merge and validate standalone add100
9. Merge and validate three-task n150
10. Parameterize and run n150 SBERT/correctness analysis
```

All ten execution steps above were completed. The retained manifest, shards, logs, standalone add100 files, validation reports, n=150 files, SBERT outputs, and correctness outputs form the audit trail for this run.
