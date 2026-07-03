# 用于 Sample-Noise Correction 对比实验的数据集选择

本文目标是从 5 篇参考文献中找出最适合本项目四类任务的数据集，并设计一个 apple-to-apple 对比框架：

```text
same dataset
same perturbation scheme
same evaluation criterion
with sample-noise correction
vs.
without sample-noise correction
```

项目四类任务为：

1. Factual question answering
2. Mathematical reasoning
3. Code generation
4. Open-ended writing / open-ended generation

## 结论总览

| 项目任务 | 五篇文献中最匹配的数据集 | 来源文献 | 是否完全匹配 | 推荐使用 |
|---|---|---|---:|---:|
| Factual QA | SQuAD V2 | PromptRobust / PromptBench | 是 | 是 |
| Mathematical reasoning | MATH / Mathematics | PromptRobust / PromptBench | 是 | 是 |
| Code generation | HumanEval | 五篇文献均未覆盖代码生成数据集；HumanEval 作为外部补充 | 否 | 是，作为外部补充 |
| Open-ended writing | Alpaca | POSIX | 较匹配 | 是 |
| Open-ended creative generation | Alternative Uses Task (AUT) | Haase et al. | 部分匹配 | 可作为 writing 的补充 |

最严格地说，如果数据集必须全部来自这 5 篇参考文章，那么项目只能完整覆盖：

```text
factual QA        -> SQuAD V2
math reasoning    -> MATH / Mathematics
open-ended writing -> Alpaca 或 AUT
```

**Code generation 在这 5 篇文章中没有合适数据集。** 如果项目必须保留 code generation 作为第四类任务，本项目选用 **HumanEval** 作为外部补充数据集。HumanEval 可以使用相同扰动方案和相同 PDR-style 评价逻辑，但不能声称数据集来自这 5 篇参考文献。

## 五篇文献的数据集映射

| 参考文献 | 使用 / 涉及的数据集 | 可匹配的项目任务 | 适合程度 | 备注 |
|---|---|---|---|---|
| PromptRobust / PromptBench | SQuAD V2 | Factual QA | 高 | 标准抽取式 / factual QA，最适合本项目 QA 任务 |
| PromptRobust / PromptBench | Mathematics / MATH | Mathematical reasoning | 高 | 最适合数学推理任务 |
| PromptRobust / PromptBench | SST-2 等分类数据集 | 不属于四类主任务 | 低 | 可用于分类 pilot，但不适合作为主任务 |
| POSIX | Alpaca | Open-ended writing / instruction following | 高 | 适合开放式生成或写作类任务 |
| POSIX | MMLU | Factual QA / knowledge QA / multiple-choice reasoning | 中 | 更偏多选知识测试，不是标准开放 QA |
| POSIX | BBH | Reasoning | 中 | 可作为 reasoning 补充，但不是专门 math reasoning |
| What Did I Do Wrong? | CB、RTE、DBPedia、WoS、TREC 等分类任务 | 不直接匹配四类主任务 | 低 | 主要支持 sensitivity / consistency 指标，不是数据集主来源 |
| Enhancing LLM Robustness to Perturbed Instructions | CoLA、QNLI、SST-2 | 分类 / NLI | 低到中 | QNLI 与 QA 有关，但本质是 NLI，不如 SQuAD V2 适合 |
| Haase et al. | Alternative Uses Task (AUT), plastic bottle | Open-ended creative writing | 中 | 适合开放式创造性生成，但任务范围较窄 |

## 推荐实验数据集组合

### 方案 A：严格基于五篇文献，覆盖三类任务

如果必须保证所有数据集都来自 5 篇参考文章，推荐：

| 任务 | 数据集 | 来源文献 | 扰动方案 | 评价准则 |
|---|---|---|---|---|
| Factual QA | SQuAD V2 | PromptRobust / PromptBench | Paraphrase、format change、information reordering、surface noise on instruction、optional context injection | PDR based on answer correctness；Sensitivity；semantic preservation |
| Math reasoning | MATH / Mathematics | PromptRobust / PromptBench | Paraphrase of instruction、format change、instruction reordering、surface noise on instruction | PDR based on final-answer correctness；Sensitivity；semantic preservation |
| Open-ended writing | Alpaca | POSIX | Paraphrase、format change、information reordering、surface noise、context injection | Sensitivity；semantic coherence；response diversity |
| Open-ended creative generation | AUT | Haase et al. | Paraphrase、formatting tweak、information order change、typo robustness | Originality；fluency；within-LLM variance |

这个方案的优点是文献来源非常干净。缺点是无法覆盖 code generation。

### 方案 B：保留四类任务，代码生成额外补充

如果项目必须保留四类任务，推荐：

| 任务 | 数据集 | 来源 | 是否来自五篇文献 | 扰动方案 | 评价准则 |
|---|---|---|---:|---|---|
| Factual QA | SQuAD V2 | PromptRobust / PromptBench | 是 | 同一套自然 perturbation | PDR based on answer correctness |
| Math reasoning | MATH / Mathematics | PromptRobust / PromptBench | 是 | 同一套自然 perturbation | PDR based on final-answer correctness |
| Code generation | HumanEval | 外部补充 | 否 | 同一套自然 perturbation，但不扰动 signature / examples | PDR based on unit-test pass rate |
| Open-ended writing | Alpaca | POSIX | 是 | 同一套自然 perturbation | Sensitivity；semantic coherence；response diversity |

这个方案更符合项目原始四类任务设计。需要在论文 / proposal 里明确说明：

```text
The code-generation dataset is added externally because none of the five reviewed prompt-perturbation papers provides a dedicated code-generation benchmark.
```

## Apple-to-apple 对比框架

为了比较 sample-noise correction 是否能更准确评价 prompt perturbation 的影响，必须保证两组实验只有一个区别：

```text
是否使用 repeated sampling 和 sample-noise correction
```

其他条件必须相同：

| 控制项 | 要求 |
|---|---|
| Dataset | corrected 和 uncorrected 使用完全相同的数据集样本 |
| Prompt | 使用同一 clean prompt |
| Perturbation | 使用同一 perturbed prompt |
| Model | 使用同一模型 |
| Decoding parameters | 使用同一 temperature、top_p、max_tokens |
| Evaluation criterion | 使用同一评价准则 |
| Sample set | corrected 和 uncorrected 从同一批 repeated generations 中计算，避免额外随机差异 |

## 每类任务的具体比较设计

### 1. Factual QA: SQuAD V2

**数据集来源**

PromptRobust / PromptBench 使用 SQuAD V2 作为 QA 类任务数据集。

**扰动方案**

使用同一套项目 perturbation：

```text
P1 paraphrase
P2 format change
P3 information reordering
P4 surface noise on instruction only
P5 optional context injection
```

注意：不扰动 passage、question、实体名、数字、日期。

**评价准则**

来自参考文献的准则：

| 准则 | 来源 |
|---|---|
| PDR / APDR | PromptRobust；Enhancing LLM Robustness |
| Sensitivity | What Did I Do Wrong |
| Semantic preservation | PromptRobust human semantic check；POSIX semantic coherence |
| Within-LLM variance | Haase et al. |

**不做 correction**

```text
Generate 1 output for clean prompt.
Generate 1 output for perturbed prompt.
Compute correctness drop / PDR.
```

**做 correction**

```text
Generate K outputs for clean prompt.
Generate K outputs for perturbed prompt.
Compute mean correctness for clean and perturbed prompts.
Estimate within-prompt variance from repeated outputs.
Check whether between-prompt difference exceeds within-prompt sample noise.
```

**比较对象**

```text
single-sample PDR
vs.
repeated-sampling / noise-aware PDR
```

### 2. Mathematical Reasoning: MATH / Mathematics

**数据集来源**

PromptRobust / PromptBench 使用 Mathematics / MATH 类数据集作为数学推理任务。

**扰动方案**

```text
P1 paraphrase of instruction
P2 output-format change
P3 instruction-order change
P4 surface noise on instruction only
```

注意：不扰动 problem statement、数字、公式、变量、单位、条件。

**评价准则**

| 准则 | 来源 |
|---|---|
| PDR / APDR | PromptRobust；Enhancing LLM Robustness |
| Sensitivity | What Did I Do Wrong |
| Semantic preservation / instruction similarity | PromptRobust；Enhancing LLM Robustness |
| Within-LLM variance | Haase et al. |

**不做 correction**

```text
Single clean output final-answer correctness
vs.
single perturbed output final-answer correctness
```

**做 correction**

```text
Repeated clean outputs final-answer correctness
vs.
repeated perturbed outputs final-answer correctness
```

**比较对象**

```text
single-sample final-answer PDR
vs.
repeated-sampling final-answer PDR
```

如果单次输出显示 perturbed prompt 错误，但 repeated clean prompt 本身也经常错，则说明原本的 perturbation effect 可能被 sample noise 夸大。

### 3. Code Generation: HumanEval

**数据集来源**

五篇参考文献中没有专门的 code generation benchmark。因此：

```text
HumanEval is introduced as an external dataset.
```

**为什么仍然可以纳入同一比较框架**

虽然数据集不来自五篇文献，但评价准则可以沿用 PromptRobust / Enhancing LLM Robustness 的 PDR 逻辑：

```text
Performance = unit-test pass rate
PDR = (Performance_clean - Performance_perturbed) / Performance_clean
```

**扰动方案**

```text
P1 paraphrase
P2 format change
P3 information reordering
P4 surface noise on natural-language instruction only
P5 optional context injection
```

注意：不扰动 function signature、example input/output、test-relevant constraints、variable names inside code blocks。

**评价准则**

| 准则 | 来源 |
|---|---|
| PDR / APDR, with performance as pass rate | PromptRobust；Enhancing LLM Robustness |
| Sensitivity | What Did I Do Wrong |
| Consistency | What Did I Do Wrong |
| Semantic preservation | PromptRobust；Enhancing LLM Robustness |
| Within-LLM variance | Haase et al. |

**不做 correction**

```text
One clean generated solution -> unit tests
One perturbed generated solution -> unit tests
Compute pass/fail difference or PDR.
```

**做 correction**

```text
K clean generated solutions -> pass rate
K perturbed generated solutions -> pass rate
Compare pass-rate drop against within-prompt pass/fail variability.
```

**比较对象**

```text
single-sample pass/fail difference
vs.
repeated-sampling pass-rate PDR
```

### 4. Open-Ended Writing: Alpaca / AUT

**数据集来源**

推荐主数据集：

```text
Alpaca from POSIX
```

可选补充：

```text
Alternative Uses Task from Haase et al.
```

Alpaca 更接近一般 open-ended instruction following；AUT 更适合 creative generation。

**扰动方案**

```text
P1 paraphrase
P2 format change
P3 information reordering
P4 surface noise
P5 context injection
```

**评价准则**

| 准则 | 来源 |
|---|---|
| Response diversity | POSIX |
| Semantic coherence | POSIX |
| Sensitivity | What Did I Do Wrong |
| Originality | Haase et al. |
| Fluency | Haase et al. |
| Within-LLM variance | Haase et al. |

**不做 correction**

```text
Generate 1 clean output.
Generate 1 perturbed output.
Compare output similarity / semantic coherence / fluency.
```

**做 correction**

```text
Generate K clean outputs.
Generate K perturbed outputs.
Estimate within-prompt output variability.
Compare between-prompt difference against within-prompt variance.
```

**比较对象**

```text
single-output drift
vs.
noise-corrected drift based on repeated outputs
```

这类任务最需要 sample-noise correction，因为 open-ended outputs 本身有较大自然变异。

## 推荐最终实验矩阵

如果项目必须保留四类任务，建议使用以下矩阵：

| Task | Dataset | Source | Perturbations | Evaluation criterion | Correction comparison |
|---|---|---|---|---|---|
| Factual QA | SQuAD V2 | PromptRobust | P1-P4，P5 optional | PDR based on answer correctness；Sensitivity | single-sample PDR vs repeated/noise-aware PDR |
| Math reasoning | MATH / Mathematics | PromptRobust | P1-P4 | PDR based on final-answer correctness；Sensitivity | single-sample PDR vs repeated/noise-aware PDR |
| Code generation | HumanEval | External, because absent from five papers | P1-P4，P5 optional | PDR based on unit-test pass rate；Sensitivity / Consistency | single pass/fail change vs repeated pass-rate PDR |
| Open-ended writing | Alpaca | POSIX | P1-P5 | Sensitivity；semantic coherence；response diversity | single-output drift vs noise-corrected drift |

如果导师或 proposal 要求“全部数据集必须来自五篇参考文献”，则应改为三类主任务：

| Task | Dataset | Source |
|---|---|---|
| Factual QA | SQuAD V2 | PromptRobust |
| Math reasoning | MATH / Mathematics | PromptRobust |
| Open-ended writing | Alpaca | POSIX |

然后把 AUT 作为 open-ended creative generation 的补充，把 code generation 标注为未来扩展。

## 推荐写法

```text
To compare prompt-perturbation evaluation with and without sample-noise correction, the study keeps the dataset, perturbation type, model, decoding parameters, and evaluation criterion fixed. For each task, the uncorrected condition estimates the perturbation effect from a single clean output and a single perturbed output. The corrected condition uses repeated generations under the same clean and perturbed prompts to estimate within-prompt sample variability, following the motivation from Haase et al. The key comparison is whether the perturbation effect measured by the same reference criterion, such as PDR, sensitivity, semantic coherence, response diversity, originality, or fluency, remains after accounting for within-prompt variability.

The dataset choices are grounded in the five reviewed papers where possible: SQuAD V2 and MATH from PromptRobust for factual QA and mathematical reasoning, and Alpaca from POSIX for open-ended writing. Code generation is not covered by the five reviewed papers, so HumanEval is added externally to retain the four-task design.
```

## 最重要的限制说明

需要明确写出：

```text
Among the five reviewed papers, no paper provides a dedicated code-generation dataset. Therefore, a strictly reference-only dataset selection cannot cover all four proposed task types. This study adds HumanEval as an external code-generation benchmark; otherwise, a reference-only experiment would need to be limited to factual QA, mathematical reasoning, and open-ended generation.
```
