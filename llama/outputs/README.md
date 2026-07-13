# Llama outputs README

本目录保存 Llama 分支的 RQ1 generation 输出与后续分析结果。之前的 Llama n50
generation 使用 Together AI serverless chat completions 完成；脚本本身使用
OpenAI-compatible chat completions 接口，prompt 文件复用主项目 `Pioneer/prompts/`
下的数据，输出写入 `Pioneer/llama/outputs/`。

历史 n50 运行配置见 `Pioneer/project_continuation.md`：

- Provider: Together AI serverless
- Endpoint: `https://api.together.ai/v1/chat/completions`
- Model: `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
- 说明：原计划的 `meta-llama/Meta-Llama-3.1-8B-Instruct` 在当时账户下不是可用的
  Together serverless endpoint，因此实际 n50 使用的是 3-8B Instruct Lite。

## 当前 n50 数据量和 repeat

已检查 `rq1_llama_*generations*.csv`，当前 n50 数据均已做 5 次 repeated
generation，`sample_id=1,2,3,4,5` 齐全。

### Original outputs

| 文件 | cases | rows | repeats |
| --- | ---: | ---: | ---: |
| `rq1_llama_original_generations_n50_factual_qa.csv` | 50 | 250 | 5 |
| `rq1_llama_original_generations_n50_math_reasoning.csv` | 50 | 250 | 5 |
| `rq1_llama_original_generations_n50_code_generation.csv` | 50 | 250 | 5 |
| `rq1_llama_original_generations_n50_open_ended_writing.csv` | 50 | 250 | 5 |

### Perturbed outputs

每个 task 的完整 perturbation 数据结构为：

`50 cases x 5 perturbation types x 5 repeats = 1250 rows`

5 类 perturbation 为：

- `paraphrasing`
- `reordering`
- `formatting_changes`
- `context_injection`
- `surface_noise`

已检查文件：

| 文件 | cases | perturbations | rows | repeats |
| --- | ---: | ---: | ---: | ---: |
| `rq1_llama_perturbed_generations_n50_factual_qa.csv` | 50 | 5 | 1250 | 5 |
| `rq1_llama_perturbed_generations_n50_factual_qa_fixed_factual.csv` | 50 | 5 | 1250 | 5 |
| `rq1_llama_perturbed_generations_n50_math_reasoning.csv` | 50 | 5 | 1250 | 5 |
| `rq1_llama_perturbed_generations_n50_math_reasoning_fixed.csv` | 50 | 5 | 1250 | 5 |
| `rq1_llama_perturbed_generations_n50_code_generation.csv` | 50 | 5 | 1250 | 5 |
| `rq1_llama_perturbed_generations_n50_open_ended_writing.csv` | 50 | 5 | 1250 | 5 |

Fixed paraphrasing 子集：

| 文件 | cases | perturbations | rows | repeats |
| --- | ---: | ---: | ---: | ---: |
| `rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv` | 50 | 1 | 250 | 5 |
| `rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv` | 50 | 1 | 250 | 5 |

检查结论：每个 original `item_id` 都刚好 5 行；每个 perturbed
`item_id + perturbation_type` 也都刚好 5 行。

## Fixed 数据优先级

后续 factual QA 和 math reasoning 的修复版分析应优先使用 fixed 文件：

- Factual QA full perturbation:
  `rq1_llama_perturbed_generations_n50_factual_qa_fixed_factual.csv`
- Factual QA paraphrasing only:
  `rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv`
- Math reasoning full perturbation:
  `rq1_llama_perturbed_generations_n50_math_reasoning_fixed.csv`
- Math reasoning paraphrasing only:
  `rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv`

不带 fixed 后缀的 factual/math perturbed 文件可以作为历史版本或对照，但不建议作为
fixed factual/math 主线分析的最终输入。

### Factual QA 旧版问题说明

`rq1_llama_perturbed_generations_n50_factual_qa.csv` 的数量和 repeat 本身是完整的：
50 cases、5 类 perturbation、每组 5 次 repeated generation，共 1250 行。

但它的 paraphrasing 条件来自旧版 factual QA paraphrasing prompts。旧 prompt 中有
7 个 item 带有 rewrite-prefix residue，因此对应到 Llama generation 中是：

- affected items: 7
- affected paraphrasing rows: 35
- affected item ids:
  `factual_qa_10550`, `factual_qa_1292`, `factual_qa_7854`,
  `factual_qa_7989`, `factual_qa_2682`, `factual_qa_9228`,
  `factual_qa_11075`

修复报告见主项目：
`Pioneer/outputs/factual_paraphrase_prompt_repair_report.md`。报告结论是 fixed
factual-QA paraphrasing prompts 已全部包含完整 `Context:` block 和干净的
`Question:` block。

因此：

- 不建议把 `rq1_llama_perturbed_generations_n50_factual_qa.csv` 作为最终 factual QA
  分析输入。
- 应优先使用 `rq1_llama_perturbed_generations_n50_factual_qa_fixed_factual.csv`。
- 如果只分析 paraphrasing，应使用
  `rq1_llama_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv`。

### Math reasoning 旧版问题说明

Llama math reasoning 与 GPT/main 的情况一致：数量和 repeat 本身完整，但旧版
paraphrasing prompt 存在内容质量问题。参考主项目
`Pioneer/outputs/math_paraphrase_prompt_repair_report.md`：

- template artifact before repair: 16 items
- ASY diagram removed before repair: 7 items
- unique affected items before repair: 18 items
- 对应到 5 次 repeated generation，旧版 Llama math 中最多会影响 90 行
  paraphrasing-condition outputs。

因此：

- `rq1_llama_perturbed_generations_n50_math_reasoning.csv` 只应视为旧版
  legacy/input-to-repair，不建议作为最终 fixed math 分析输入。
- 应优先使用 `rq1_llama_perturbed_generations_n50_math_reasoning_fixed.csv`。
- 如果只分析 paraphrasing，应使用
  `rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv`。

额外复核发现：fixed math 文件已修复报告中定义的 template artifact 和 ASY 删除问题；
但 `math_reasoning_5201` 的 fixed paraphrasing prompt 仍保留一句
`Rewrite the prompt while maintaining its core meaning`，因此它是一个 residual wording
artifact。这个 residual 同时存在于 GPT/main fixed math 和 Llama fixed math，因为两者复用
同一批 fixed prompt。若后续需要更严格的 fixed math 主线，应单独再修复该 item 并重跑
对应 paraphrasing generation。

## add100 目标

下一步需要给 Llama 也增加 100 个 case。和 GPT add100 一致，本批只增加三类任务：

- `factual_qa`
- `math_reasoning`
- `code_generation`

不增加 `open_ended_writing`。

主项目中 add100 prompt 已准备好：

| prompt 文件 | rows | cases | 说明 |
| --- | ---: | ---: | --- |
| `prompts/rq1_sampled_original_prompts_add100_three_task.csv` | 300 | 300 | 三类任务各 100 个 original case |
| `prompts/rq1_formal_perturbed_prompts_add100_factual_qa.csv` | 500 | 100 | factual QA，5 类 perturbation |
| `prompts/rq1_formal_perturbed_prompts_add100_math_reasoning.csv` | 500 | 100 | math reasoning，5 类 perturbation |
| `prompts/rq1_formal_perturbed_prompts_add100_code_generation.csv` | 500 | 100 | code generation，5 类 perturbation |

Llama add100 预期输出：

- Original: `300 cases x 5 repeats = 1500 rows`
- Perturbed: `300 cases x 5 perturbation types x 5 repeats = 7500 rows`

推荐正式合并输出文件名：

- `rq1_llama_original_generations_add100_three_task.csv`
- `rq1_llama_perturbed_generations_add100_three_task.csv`
- `add100_generation_validation_report.md`

## 推荐运行流程

先设置 Together AI endpoint。为了和既有 Llama n50 数据保持可比性，推荐继续使用
之前实际使用过的 Together 模型：

```powershell
$env:LLAMA_API_URL = "https://api.together.ai/v1/chat/completions"
$env:LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
$env:LLAMA_API_KEY = (Get-Content .\together_api.txt -Raw).Trim()
```

如果后续改用本地 vLLM 或其他 OpenAI-compatible Llama 服务，可以改回对应 endpoint；
但会引入模型/服务来源差异，不建议和 Together n50 直接混用。

### 1. 切 5 个 shard

```powershell
python .\Pioneer\src\61_split_add100_generation_shards.py `
  --original-inputs prompts\rq1_sampled_original_prompts_add100_three_task.csv `
  --perturbed-inputs `
    prompts\rq1_formal_perturbed_prompts_add100_factual_qa.csv `
    prompts\rq1_formal_perturbed_prompts_add100_math_reasoning.csv `
    prompts\rq1_formal_perturbed_prompts_add100_code_generation.csv `
  --output-dir llama\tmp_parallel\add100 `
  --shards 5 `
  --expected-items-per-task 100
```

切分后每路应为：

- original prompt rows: 60
- perturbed prompt rows: 300
- expected original generation rows: 300
- expected perturbed generation rows: 1500

### 2. 生成 original outputs

每个 shard 跑：

```powershell
python .\Pioneer\llama\src\07_generate_rq1_outputs_llama.py `
  --input llama\tmp_parallel\add100\original_prompts_shard1.csv `
  --output llama\outputs\add100_shards\original_generations_shard1.csv `
  --samples 5
```

将 `shard1` 改为 `shard2` 到 `shard5`，可 5 路并行。每个 shard 预期 300 行。

### 3. 生成 perturbed outputs

每个 shard 跑：

```powershell
python .\Pioneer\llama\src\11_generate_rq1b_perturbed_outputs_llama.py `
  --input llama\tmp_parallel\add100\perturbed_prompts_shard1.csv `
  --output llama\outputs\add100_shards\perturbed_generations_shard1.csv `
  --samples 5
```

将 `shard1` 改为 `shard2` 到 `shard5`，可 5 路并行。每个 shard 预期 1500 行。

### 4. Merge and validate

```powershell
python .\Pioneer\src\62_merge_validate_add100_outputs.py `
  --original-patterns llama\outputs\add100_shards\original_generations_shard*.csv `
  --perturbed-patterns llama\outputs\add100_shards\perturbed_generations_shard*.csv `
  --original-output llama\outputs\rq1_llama_original_generations_add100_three_task.csv `
  --perturbed-output llama\outputs\rq1_llama_perturbed_generations_add100_three_task.csv `
  --report llama\outputs\add100_generation_validation_report.md `
  --expected-items-per-task 100 `
  --expected-samples 5
```

校验通过时应满足：

- original rows = 1500
- perturbed rows = 7500
- no duplicate keys
- no empty outputs
- original and perturbed item sets match
- every prompt has `sample_id=1,2,3,4,5`

## 断点续跑

Llama generation 脚本会读取已有输出并跳过已存在 key：

- original key: `item_id + sample_id`
- perturbed key: `item_id + perturbation_type + sample_id`

如果某个 shard 中途失败，直接重跑同一条命令即可补齐剩余行，不会覆盖已完成结果。

## add100 五路并行正式执行计划

目标是在 Llama 分支中再增加 100 个 case。为保持和 GPT/main add100 一致，本轮只做
三类任务：

- `factual_qa`
- `math_reasoning`
- `code_generation`

不做 `open_ended_writing`。最终目标数据量为：

- original generations: 300 cases x 5 repeats = 1500 rows
- perturbed generations: 300 cases x 5 perturbations x 5 repeats = 7500 rows

### 执行原则

采用 5 路 shard 并行，但建议分两个阶段跑：

1. 先跑 original 5 路，每路 300 rows，总计 1500 rows。
2. original 全部完成并计数确认后，再跑 perturbed 5 路，每路 1500 rows，总计 7500 rows。

不建议 original 和 perturbed 同时开 10 路。perturbed prompt 更长，接口更容易出现超时、
限流或 provider 端临时错误；分阶段跑更容易定位失败 shard，也更容易续跑。

### Step 0: preflight check

确认 Together AI endpoint 和模型设置。为了和既有 Llama n50 数据保持可比性，推荐使用
之前实际采用的 Together serverless 模型：

```powershell
$env:LLAMA_API_URL = "https://api.together.ai/v1/chat/completions"
$env:LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
$env:LLAMA_API_KEY = (Get-Content .\together_api.txt -Raw).Trim()
```

不要在同一批 add100 中混用 Together serverless、local vLLM 或其他 provider。若必须更换
provider/model，应单独命名输出文件，并在 validation report 或 README 中标注。

确认 add100 prompt 数量：

```powershell
$files = @(
  "prompts\rq1_sampled_original_prompts_add100_three_task.csv",
  "prompts\rq1_formal_perturbed_prompts_add100_factual_qa.csv",
  "prompts\rq1_formal_perturbed_prompts_add100_math_reasoning.csv",
  "prompts\rq1_formal_perturbed_prompts_add100_code_generation.csv"
)
foreach($f in $files){
  $rows = Import-Csv ".\Pioneer\$f"
  $cases = @($rows | Select-Object -ExpandProperty item_id -Unique).Count
  "$f rows=$(@($rows).Count) cases=$cases"
}
```

预期结果：

- original three-task prompt: 300 rows, 300 cases
- each perturbed task prompt: 500 rows, 100 cases

### Step 1: split into 5 shards

```powershell
python .\Pioneer\src\61_split_add100_generation_shards.py `
  --original-inputs prompts\rq1_sampled_original_prompts_add100_three_task.csv `
  --perturbed-inputs `
    prompts\rq1_formal_perturbed_prompts_add100_factual_qa.csv `
    prompts\rq1_formal_perturbed_prompts_add100_math_reasoning.csv `
    prompts\rq1_formal_perturbed_prompts_add100_code_generation.csv `
  --output-dir llama\tmp_parallel\add100 `
  --shards 5 `
  --expected-items-per-task 100
```

预期 shard manifest：

- each original shard: 60 prompt rows, expected 300 generation rows
- each perturbed shard: 300 prompt rows, expected 1500 generation rows

### Step 2: run original 5-way parallel

```powershell
$outDir = ".\Pioneer\llama\outputs\add100_shards"
$logDir = ".\Pioneer\llama\outputs\add100_logs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

for($i = 1; $i -le 5; $i++){
  Start-Process -FilePath "python" `
    -ArgumentList @(
      ".\Pioneer\llama\src\07_generate_rq1_outputs_llama.py",
      "--input", "llama\tmp_parallel\add100\original_prompts_shard$i.csv",
      "--output", "llama\outputs\add100_shards\original_generations_shard$i.csv",
      "--samples", "5"
    ) `
    -WorkingDirectory "D:\pioneer_kayley_llm" `
    -WindowStyle Hidden `
    -RedirectStandardOutput ".\Pioneer\llama\outputs\add100_logs\original_shard$i.out.log" `
    -RedirectStandardError ".\Pioneer\llama\outputs\add100_logs\original_shard$i.err.log"
}
```

进度统计：

```powershell
$total = 0
for($i = 1; $i -le 5; $i++){
  $p = ".\Pioneer\llama\outputs\add100_shards\original_generations_shard$i.csv"
  if(Test-Path $p){ $n = @(Import-Csv $p).Count } else { $n = 0 }
  $total += $n
  "original shard$i`t$n/300"
}
"original total`t$total/1500`t$([math]::Round($total / 1500 * 100, 1))%"
```

Original 完成标准：5 个 shard 都是 `300/300`，总计 `1500/1500`。

### Step 3: run perturbed 5-way parallel

Original 完成后再启动 perturbed：

```powershell
$outDir = ".\Pioneer\llama\outputs\add100_shards"
$logDir = ".\Pioneer\llama\outputs\add100_logs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

for($i = 1; $i -le 5; $i++){
  Start-Process -FilePath "python" `
    -ArgumentList @(
      ".\Pioneer\llama\src\11_generate_rq1b_perturbed_outputs_llama.py",
      "--input", "llama\tmp_parallel\add100\perturbed_prompts_shard$i.csv",
      "--output", "llama\outputs\add100_shards\perturbed_generations_shard$i.csv",
      "--samples", "5"
    ) `
    -WorkingDirectory "D:\pioneer_kayley_llm" `
    -WindowStyle Hidden `
    -RedirectStandardOutput ".\Pioneer\llama\outputs\add100_logs\perturbed_shard$i.out.log" `
    -RedirectStandardError ".\Pioneer\llama\outputs\add100_logs\perturbed_shard$i.err.log"
}
```

进度统计：

```powershell
$total = 0
for($i = 1; $i -le 5; $i++){
  $p = ".\Pioneer\llama\outputs\add100_shards\perturbed_generations_shard$i.csv"
  if(Test-Path $p){ $n = @(Import-Csv $p).Count } else { $n = 0 }
  $total += $n
  "perturbed shard$i`t$n/1500"
}
"perturbed total`t$total/7500`t$([math]::Round($total / 7500 * 100, 1))%"
```

Perturbed 完成标准：5 个 shard 都是 `1500/1500`，总计 `7500/7500`。

### Step 4: failure handling

如果某个 shard 停止增长：

1. 查看对应 error log：

```powershell
Get-Content .\Pioneer\llama\outputs\add100_logs\perturbed_shard3.err.log -Tail 40
```

2. 直接重跑同一 shard。脚本会自动跳过已有 key，只补剩余行：

```powershell
python .\Pioneer\llama\src\11_generate_rq1b_perturbed_outputs_llama.py `
  --input llama\tmp_parallel\add100\perturbed_prompts_shard3.csv `
  --output llama\outputs\add100_shards\perturbed_generations_shard3.csv `
  --samples 5
```

Original shard 失败时也一样重跑对应 original 命令。

### Step 5: merge and validate

```powershell
python .\Pioneer\src\62_merge_validate_add100_outputs.py `
  --original-patterns llama\outputs\add100_shards\original_generations_shard*.csv `
  --perturbed-patterns llama\outputs\add100_shards\perturbed_generations_shard*.csv `
  --original-output llama\outputs\rq1_llama_original_generations_add100_three_task.csv `
  --perturbed-output llama\outputs\rq1_llama_perturbed_generations_add100_three_task.csv `
  --report llama\outputs\add100_generation_validation_report.md `
  --expected-items-per-task 100 `
  --expected-samples 5
```

最终验收标准：

- `rq1_llama_original_generations_add100_three_task.csv`: 1500 rows
- `rq1_llama_perturbed_generations_add100_three_task.csv`: 7500 rows
- `add100_generation_validation_report.md`: PASS
- duplicate keys: none
- empty outputs: none
- original and perturbed item sets: match
- every original `item_id` has 5 repeats
- every perturbed `item_id + perturbation_type` has 5 repeats
