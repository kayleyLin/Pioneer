# Pioneer/outputs 数据目录说明

本目录主要保存 GPT/main 分支的 RQ1 generation、SBERT drift 分析、fixed factual QA follow-up、math follow-up 和若干中间诊断结果。目录里文件较多，且不同日期/不同实验阶段的输出混在一起；后续读数据时建议按本 README 的“主线文件”优先级使用。

## 快速结论

如果你只想找 GPT/main 生成的 50 条 factual QA 数据，优先看这几类文件：

| 用途 | 推荐文件 | 行数 | 说明 |
|---|---|---:|---|
| 原始 factual QA 生成 | `rq1_formal_original_generations_n50_factual_qa.csv` | 250 | 50 个 item，每个 item 5 个 original outputs |
| factual QA 全扰动生成 | `rq1_formal_perturbed_generations_n50_factual_qa_fixed.csv` | 1250 | 修复后的 factual QA perturbed generations，包含多种 perturbation |
| fixed paraphrasing 生成 | `rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv` | 250 | 50 个 item，每个 item 5 个 paraphrasing outputs |
| item 级 drift 主表 | `factual_paraphrase_item_table_fixed_factual.csv` | 50 | fixed factual paraphrase 的 item-level 主表 |
| SBERT item 级 drift | `sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv` | 50 | original vs paraphrase 的 SBERT drift |
| cue/prompt 诊断 | `factual_paraphrase_cue_metrics_fixed_factual.csv` | 50 | question/cue/context 保留程度 |
| correctness/长度诊断 | `factual_paraphrase_correctness_by_item_fixed_factual.csv` | 50 | reference token-F1、containment、output length delta |
| text-feature driver 主表 | `factual_text_feature_base_fixed_factual.csv` | 50 | prompt/output edit distance、长度、scope/style proxy |
| text-feature 相关性 | `factual_text_feature_driver_correlations_fixed_factual.csv` | 20 | 20 个 feature 与 `noise_corrected_drift` 的 Spearman 相关 |
| text-feature 摘要 | `factual_text_feature_driver_summary_fixed_factual.md` | - | GPT/main text-level driver 分析总结 |

## 目录里的数据分组

### 1. RQ1 formal generation 主数据

这些文件是 GPT/main 的 n=50 formal generation 数据。命名中的 `n50` 表示每个 task 选了 50 个 item；通常每个 item 有 5 个 sampled outputs。

Original generations：

- `rq1_formal_original_generations_n50_factual_qa.csv`
- `rq1_formal_original_generations_n50_math_reasoning.csv`
- `rq1_formal_original_generations_n50_code_generation.csv`
- `rq1_formal_original_generations_n50_open_ended_writing.csv`

Perturbed generations：

- `rq1_formal_perturbed_generations_n50_factual_qa_fixed.csv`
- `rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv`
- `rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv`
- `rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv`
- `rq1_formal_perturbed_generations_n50_code_generation.csv`
- `rq1_formal_perturbed_generations_n50_open_ended_writing.csv`

注意：factual QA perturbed generation 只保留修复后的正确版本：

- `rq1_formal_perturbed_generations_n50_factual_qa_fixed.csv`
- `rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv`

`paraphrasing_fixed` 只保留 factual QA paraphrasing 条件，因此是 250 行；`fixed.csv` 包含 factual QA 的多种 perturbation，因此是 1250 行。

清理记录（2026-07-12）：

- 已删除 `rq1_formal_perturbed_generations_n50_factual_qa.csv`。这是早期有问题的 factual QA perturbed generation，不再作为 fixed factual QA 分析输入。
- 已删除 `rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed_raw_with_duplicates.csv`。这是修复过程中的 raw intermediate，包含 duplicates，不应作为最终分析输入。
- 当前 factual QA perturbed generation 主线只保留 `_fixed.csv` 和 `_paraphrasing_fixed.csv` 两个正确版本。

math reasoning 也存在旧版和修复版并存的情况。最终 fixed math 分析应优先使用：

- `rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv`
- `rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv`

`rq1_formal_perturbed_generations_n50_math_reasoning.csv` 是旧版未修复 generation。它目前保留在目录中，原因是 `src/51_merge_fixed_math_paraphrase_generations.py` 需要把它作为 `original_full` 输入，再与 `math_reasoning_paraphrasing_fixed` 合并生成 `math_reasoning_fixed`。因此它只应被视为 legacy/input-to-repair，不应作为当前 fixed math 的最终分析输入。

### 2. SBERT drift 结果

这些文件把 generation 输出转换成 similarity/drift 指标。

RQ1 n=50 主结果：

- `sbert_rq1_n50_baseline_by_item.csv`
- `sbert_rq1_n50_baseline_by_task.csv`
- `sbert_rq1_n50_perturbation_effects_by_item.csv`
- `sbert_rq1_n50_perturbation_summary.csv`
- `sbert_rq1_n50_heatmap_noise_corrected_drift.csv`
- `sbert_rq1_n50_uncorrected_heatmap_drift.csv`

fixed factual QA 后的重算结果：

- `sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv`
- `sbert_rq1_n50_perturbation_effects_by_item_fixed_factual.csv`
- `sbert_rq1_n50_perturbation_summary_fixed_factual.csv`
- `sbert_rq1_n50_heatmap_noise_corrected_drift_fixed_factual.csv`
- `sbert_rq1_n50_uncorrected_heatmap_drift_fixed_factual.csv`

如果研究 fixed factual QA paraphrasing，请优先用带 `_fixed_factual` 的版本。

### 3. fixed factual QA follow-up

这些文件专门分析 factual QA paraphrasing 修复后仍然存在的 residual drift。

主表和 case 表：

- `factual_paraphrase_item_table_fixed_factual.csv`
- `factual_paraphrase_case_candidates_fixed_factual.csv`
- `factual_paraphrase_case_table_fixed_factual.md`
- `factual_paraphrase_case_table_manual_review_fixed_factual.md`

cue/prompt 诊断：

- `factual_paraphrase_cue_metrics_fixed_factual.csv`
- `factual_paraphrase_cue_correlations_fixed_factual.csv`
- `factual_paraphrase_cue_regressions_fixed_factual.csv`
- `factual_paraphrase_cue_analysis_fixed_factual.md`

correctness/长度诊断：

- `factual_paraphrase_correctness_by_item_fixed_factual.csv`
- `factual_paraphrase_correctness_summary_fixed_factual.md`

text-feature driver 分析：

- `factual_text_feature_base_fixed_factual.csv`
- `factual_text_feature_driver_correlations_fixed_factual.csv`
- `factual_text_feature_driver_regressions_fixed_factual.csv`
- `factual_text_feature_driver_summary_fixed_factual.md`

这些文件支持的主要结论是：fixed factual QA paraphrase drift 更强地关联于 generated-output divergence、output length expansion 和 reference token-F1/compactness loss，而不是 reference answer 的简单消失。

### 4. math follow-up

这些文件属于 math paraphrase / fixed math paraphrase 的分析，不是 factual QA 主线。不要和 GPT factual QA 50 条数据混用。

math 的文件也有一条修复链路：

- 旧版输入：`rq1_formal_perturbed_generations_n50_math_reasoning.csv`
- 修复后的 paraphrasing-only 输出：`rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv`
- 修复后的全扰动输出：`rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv`
- 修复报告：`math_paraphrase_prompt_repair_report.md`

当前 fixed math 分析应使用 `_fixed.csv` 和 `_paraphrasing_fixed.csv`。旧版 `rq1_formal_perturbed_generations_n50_math_reasoning.csv` 只保留给重跑修复/merge 脚本使用，不应作为最终分析输入。

常见文件包括：

- `math_paraphrase_driver_by_item.csv`
- `math_paraphrase_driver_correlations.csv`
- `math_paraphrase_explanation.md`
- `math_fixed_paraphrase_driver_by_item.csv`
- `math_fixed_paraphrase_driver_correlations.csv`
- `math_fixed_paraphrase_reason.md`
- `math_internal_case_selection.csv`
- `math_internal_logprob_probe_by_item.csv`
- `math_internal_trajectory_comparison.csv`

这些文件用于解释 math reasoning paraphrase drift、内部 logprob probe、trajectory comparison 等。

### 5. Llama fallback/copy 文件

本目录里有少量 `llama_` 前缀文件：

- `llama_factual_text_feature_base_fixed_factual.csv`
- `llama_factual_text_feature_driver_correlations_fixed_factual.csv`
- `llama_factual_text_feature_driver_regressions_fixed_factual.csv`
- `llama_factual_text_feature_driver_summary_fixed_factual.md`

这些不是 GPT/main 数据。它们是 Llama branch 的 text-feature driver 输出，因为运行时写入 `llama/outputs/` 失败或使用 fallback，所以被写到了本目录并带有 `llama_` 前缀。

GPT/main 分析不要误用这些文件；跨模型比较时才使用。

### 6. 早期 pilot / calibration / legacy 文件

这些文件是早期实验、校准或旧版 RQ1 输出，通常不作为 fixed factual QA 主线数据：

- `generations.csv`
- `rq1_generations.csv`
- `rq1_calibration_generations.csv`
- `rq1_temperature_pilot_generations.csv`
- `rq1b_pilot_*.csv`
- `rq1_noise_baseline_*.csv`
- `rq1_real_noise_*.csv`
- `rq1_surface_noise_*`
- `sbert_rq1a_*`
- `sbert_rq1b_*`

除非要复查早期实验流程，否则建议优先使用 `rq1_formal_*` 和 `sbert_rq1_n50_*` 系列。

## 推荐读取顺序：fixed factual QA 50 条

如果目标是复现或解释 GPT/main factual QA paraphrasing 的 50 条结果，建议按这个顺序读：

1. `rq1_formal_original_generations_n50_factual_qa.csv`

   查看 50 个 factual QA item 在 original prompt 下的 5 次输出。

2. `rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed.csv`

   查看同一批 50 个 item 在 fixed paraphrasing prompt 下的 5 次输出。

3. `sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item.csv`

   查看每个 item 的 baseline similarity、perturbation similarity、uncorrected drift 和 noise-corrected drift。

4. `factual_paraphrase_item_table_fixed_factual.csv`

   把 prompt、question、context、reference answer 和 drift 合并到 item 级别。

5. `factual_paraphrase_cue_metrics_fixed_factual.csv`

   检查 question paraphrase 是否保留了 cue、entity、number、context overlap。

6. `factual_paraphrase_correctness_by_item_fixed_factual.csv`

   检查 reference token-F1、containment rate 和 output length delta。

7. `factual_text_feature_base_fixed_factual.csv`

   查看最终 text-feature driver 表，包括 prompt edit distance、output edit distance、length ratio、scope/style proxy。

8. `factual_text_feature_driver_summary_fixed_factual.md`

   直接读分析总结和主要结论。

## 关键字段解释

常用 join key：

- `item_id`：跨 generation、SBERT、cue、correctness、text-feature 表的主要连接键。
- `sample_id`：同一个 item 的第几次 sampled output。
- `perturbation_type`：扰动类型，例如 paraphrasing。

常用 drift 字段：

- `baseline_similarity`：original condition 内部输出相似度。
- `perturbation_similarity`：original output 与 perturbed output 的相似度。
- `uncorrected_drift`：未扣除 baseline noise 的漂移。
- `noise_corrected_drift`：扣除 baseline sample noise 后的漂移，后续分析的主指标。

常用 factual/text feature 字段：

- `factual_score_delta`：paraphrase condition 的 reference token-F1 减去 original condition 的 reference token-F1。负值表示 paraphrase 后 reference overlap 下降。
- `containment_rate_delta`：reference answer containment rate 的变化。这个字段在 fixed factual QA 中解释力较弱。
- `output_length_delta_tokens`：paraphrase 输出长度减去 original 输出长度。
- `mean_output_token_edit_distance_norm`：original outputs 与 paraphrase outputs 的平均 token-level edit distance。
- `question_token_edit_distance_norm`：original question 与 paraphrased question 的 token-level edit distance。
- `cue_disruption`：question cue disruption 的 proxy。

## 不建议优先使用的旧/中间文件

以下文件不是错的，但通常不是当前 fixed factual QA 主线的最终版本：

- `factual_paraphrase_item_table.csv`：早于 `_fixed_factual` 的版本。
- 不带 `_fixed_factual` 的 `sbert_rq1_n50_*`：用于原始 RQ1 n=50 结果，不是 fixed factual QA 修复后的最终事实问答解释表。

已清理的旧/中间 factual QA generation 文件：

- `rq1_formal_perturbed_generations_n50_factual_qa.csv`
- `rq1_formal_perturbed_generations_n50_factual_qa_paraphrasing_fixed_raw_with_duplicates.csv`

## 一句话说明

`Pioneer/outputs` 是 GPT/main 输出与分析结果的总仓库；研究“GPT 生成的 50 条 fixed factual QA paraphrasing 数据”时，应优先使用 `rq1_formal_*_n50_factual_qa*fixed*`、`sbert_rq1_n50_*fixed_factual*`、`factual_paraphrase_*_fixed_factual*` 和 `factual_text_feature_*_fixed_factual*` 这一条数据链。

## add100 three-task repeated generation

2026-07-12 已为新增的 100 个 case 补齐 5 次 repeated generation。该批次只包含
`factual_qa`、`math_reasoning`、`code_generation` 三类任务，不包含
`open_ended_writing`。

正式合并输出：

- `rq1_formal_original_generations_add100_three_task.csv`：300 个 original
  cases x 5 repeats = 1500 行。
- `rq1_formal_perturbed_generations_add100_three_task.csv`：300 个 cases x
  5 perturbation types x 5 repeats = 7500 行。
- `add100_generation_validation_report.md`：merge/validate 报告，确认无重复 key、
  无空输出、original 与 perturbed item set 一致，且 `sample_id=1,2,3,4,5` 齐全。

中间 shard 输出保存在 `add100_shards/`；运行日志保存在 `add100_logs/`。生成过程中
遇到过 OpenAI 偶发返回 `The model is not available` 的 404，已将该情况纳入重试逻辑，
并通过断点续跑补齐所有 shard。

### add100 paraphrasing repair record

2026-07-13 复查发现，最初的 add100 generation 完成时间早于 add100 factual/math
prompt 修复时间，因此旧版 `rq1_formal_perturbed_generations_add100_three_task.csv`
虽然行数完整，但 factual QA 和 math reasoning 的 paraphrasing 条件使用了修复前
prompt。该问题已修复。

本次修复重跑并替换：

- add100 factual QA paraphrasing：100 items x 5 repeats = 500 rows。
- add100 math reasoning paraphrasing：100 items x 5 repeats = 500 rows。

替换后重新校验 `rq1_formal_perturbed_generations_add100_three_task.csv`：

- rows = 7500
- duplicate keys = 0
- empty outputs = 0
- prompt mismatch against current `prompts/` files = 0
- factual QA paraphrasing malformed `Context:`/`Question:` prompts = 0
- math paraphrasing template artifacts = 0
- math paraphrasing ASY diagram deletion cases = 0

辅助文件：

- repair prompts: `paraphrase_repair_prompts/`
- repair outputs: `paraphrase_repair_outputs/`
- repair logs: `paraphrase_repair_logs/`
- validation report: `add100_generation_validation_report.md`

同次修复还重跑并替换了 n50 fixed math 中 `math_reasoning_5201` 的 5 行
paraphrasing generation。替换后：

- `rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv` prompt mismatch = 0
- `rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv` prompt mismatch = 0
- math paraphrasing template artifacts = 0

报告见 `n50_math_5201_repair_validation_report.md`。
