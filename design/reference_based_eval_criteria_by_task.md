# 基于五篇参考文献的项目评价准则整理

本文目标是从 5 篇参考文献中提取可用于本项目四类任务的评价准则。这里不引入新的评价指标，只整理参考文献已有的评价思想，并说明如何用于 apple-to-apple 比较：

```text
clean prompt repeated outputs
vs.
perturbed prompt repeated outputs
```

项目当前关注 similarity 和 correctness。根据参考文献，这两个维度可以分别对应到：

- Similarity / drift：POSIX 的 sensitivity 思路、What Did I Do Wrong 的 sensitivity / consistency、Haase et al. 的 output variability / variance decomposition。
- Correctness / task performance：PromptRobust 和 Enhancing LLM Robustness 使用的 performance drop / PDR，以及 What Did I Do Wrong 使用的 Micro-F1。

## 五篇文献中的评价准则总表

| 文献 | 原文使用的评价准则 | 是否需要 ground truth | 是否需要 token probability | 适合本项目的部分 |
|---|---|---:|---:|---|
| PromptRobust / PromptBench | Performance Drop Rate (PDR)、Average PDR (APDR)、human semantic preservation check、attention visualization、transferability analysis | PDR 需要 | 不需要 | PDR / APDR 可用于有标准答案任务；human semantic preservation 可用于判断扰动是否保持原任务意图 |
| POSIX | POSIX score，即同一 response 在不同 intent-aligned prompts 下的 log-likelihood 相对变化；同时分析 response diversity、entropy、semantic coherence、confidence variance | 不一定 | 需要 | sensitivity 思路适合本项目，但严格 POSIX 需要 token log-likelihood |
| What Did I Do Wrong? | Sensitivity、Consistency、Micro-F1 | Micro-F1 需要 | 不需要 | sensitivity / consistency 适合黑盒稳定性评价；Micro-F1 适合分类或可离散评分任务 |
| Enhancing LLM Robustness to Perturbed Instructions | PDR；clean instruction 与 perturbed instruction 的 embedding cosine similarity；semantic similarity 与 PDR 的关系 | PDR 需要 | 不需要 | PDR 适合 QA、math、code；instruction semantic similarity 可作为扰动有效性检查 |
| Within-Model vs Between-Prompt Variability | Originality、Fluency、linear mixed-effects variance decomposition；model effect、prompt effect、model-prompt interaction、within-LLM sampling variance | 不需要唯一答案 | 不需要 | repeated sampling 和 within-LLM variance 是 sample-noise calibration 的直接依据；originality / fluency 可用于 open-ended writing |

## 项目四类任务的推荐评价准则

### 1. Factual QA

**任务目标**

Factual QA 的输出应该回答给定问题。它有 ground truth answer，因此可以使用参考文献中的 performance-based 评价。

**适合采用的参考文献准则**

| 准则 | 来源文献 | 如何用于 factual QA | 用途 |
|---|---|---|---|
| PDR / APDR | PromptRobust；Enhancing LLM Robustness | 比较 clean prompt 与 perturbed prompt 下 QA correctness 的下降比例 | 衡量扰动是否降低任务表现 |
| Sensitivity | What Did I Do Wrong | 观察同一 QA sample 在不同 prompt variants 下答案分布是否变化 | 衡量输出稳定性 |
| Semantic coherence / semantic preservation | POSIX；PromptRobust human evaluation | 检查扰动 prompt 是否仍保持原问题意图；也可检查答案语义是否仍对应原答案 | 防止扰动改变任务本身 |
| Within-LLM variance | Haase et al. | 对 clean 和 perturbed prompt 都重复采样，估计同一 prompt 下的自然输出波动 | 用于 sample-noise baseline |

**Apple-to-apple 比较方式**

对 factual QA，最直接的参考文献式比较是：

```text
clean correctness
vs.
perturbed correctness
```

然后用 PromptRobust / Enhancing LLM Robustness 的 PDR 形式报告：

```text
PDR = (Performance_clean - Performance_perturbed) / Performance_clean
```

如果加入 sample-noise 校准，应保持同一评价准则，只是把单次 performance 改成 repeated-sampling 下的平均 performance：

```text
PDR_repeated = (MeanPerformance_clean - MeanPerformance_perturbed) / MeanPerformance_clean
```

这里的核心不是发明新指标，而是用 Haase et al. 的 repeated sampling 思路让 PDR 估计更稳。

**推荐主指标**

```text
PDR / APDR + Sensitivity + within-prompt sample variance
```

### 2. Mathematical Reasoning

**任务目标**

Math reasoning 有明确 final answer，因此可以采用 performance drop 类评价。但数学输出通常包含推理过程，所以还应区分 final-answer correctness 和 reasoning/output variability。

**适合采用的参考文献准则**

| 准则 | 来源文献 | 如何用于 math reasoning | 用途 |
|---|---|---|---|
| PDR / APDR | PromptRobust；Enhancing LLM Robustness | 比较 clean 与 perturbed prompt 下 final-answer correctness 的下降 | 衡量扰动是否导致解题失败 |
| Sensitivity | What Did I Do Wrong | 同一道题在多个 prompt variants 下 final answer 或 answer pattern 是否变化 | 衡量 prompt sensitivity |
| Semantic preservation / instruction similarity | PromptRobust；Enhancing LLM Robustness | 检查 perturbation 是否只改变 instruction 表达，而没有改变数学题条件 | 保证任务未被改写 |
| Within-LLM sampling variance | Haase et al. | 对同一 math prompt 多次生成，估计自然推理路径和答案波动 | 校准 sample noise |

**Apple-to-apple 比较方式**

Math 最适合使用 PDR，因为它有客观答案：

```text
clean final-answer accuracy
vs.
perturbed final-answer accuracy
```

sample-noise 校准时，不改变评价准则，只改变估计方式：

```text
clean prompt:      repeated final-answer correctness
perturbed prompt:  repeated final-answer correctness
```

然后比较：

```text
single-sample PDR
vs.
repeated-sampling PDR
```

如果 repeated-sampling 后 PDR 明显缩小，说明原来的单次 prompt perturbation effect 可能混入了 sampling noise。

**推荐主指标**

```text
PDR / APDR + Sensitivity + within-prompt sample variance
```

### 3. Code Generation

**任务目标**

Code generation 不在五篇参考文献中被直接覆盖，但它和 factual QA / math 一样有 objective correctness：代码能否通过测试。因此可借用 PromptRobust 和 Enhancing LLM Robustness 的 performance drop 评价逻辑。

**适合采用的参考文献准则**

| 准则 | 来源文献 | 如何用于 code generation | 用途 |
|---|---|---|---|
| PDR / APDR | PromptRobust；Enhancing LLM Robustness | 将 performance 定义为 pass rate，然后比较 clean 与 perturbed prompt 下 pass rate 下降 | 衡量扰动是否降低代码正确性 |
| Sensitivity | What Did I Do Wrong | 同一 coding task 在不同 prompt variants 下输出代码行为或通过率是否变化 | 衡量 prompt sensitivity |
| Consistency | What Did I Do Wrong | 同一类 coding problems 在 prompt variants 下是否表现出相似稳定性 | 衡量跨样本稳定性 |
| Semantic preservation / instruction similarity | PromptRobust；Enhancing LLM Robustness | 检查扰动是否没有改变函数签名、输入输出要求或约束 | 保证 perturbed prompt 仍是同一任务 |
| Within-LLM sampling variance | Haase et al. | 同一 coding prompt 多次生成，估计自然代码变体和 pass/fail 波动 | 校准 sample noise |

**Apple-to-apple 比较方式**

Code generation 可以用参考文献中的 PDR 逻辑，把 performance 替换成 pass rate：

```text
Performance = unit-test pass rate
```

于是：

```text
PDR_code = (PassRate_clean - PassRate_perturbed) / PassRate_clean
```

这不是创造新准则，而是沿用 PromptRobust / Enhancing LLM Robustness 的 PDR 框架，只是把任务 performance 换成代码生成领域的 objective correctness。

sample-noise 校准后，比较方式为：

```text
single-generation pass/fail change
vs.
repeated-sampling pass-rate change
```

如果单次生成显示 perturbed prompt 失败、clean prompt 成功，但 repeated sampling 发现 clean prompt 本身也有较高失败率，则扰动影响应被重新解释为 sample noise 或不稳定生成，而不是纯 prompt perturbation effect。

**推荐主指标**

```text
PDR based on pass rate + Sensitivity + Consistency + within-prompt sample variance
```

### 4. Open-Ended Writing / Open-Ended Generation

**任务目标**

Open-ended writing 没有唯一标准答案，因此不适合主要使用 accuracy 或 PDR。它更适合参考 POSIX、What Did I Do Wrong 和 Haase et al. 的稳定性、语义一致性和方差分解思路。

**适合采用的参考文献准则**

| 准则 | 来源文献 | 如何用于 open-ended writing | 用途 |
|---|---|---|---|
| Response diversity | POSIX | 比较 clean 与 perturbed prompt 下输出是否变得更多样或更分散 | 衡量输出分布变化 |
| Semantic coherence | POSIX | 检查输出是否仍然语义连贯、符合原任务意图 | 衡量输出质量 |
| Sensitivity | What Did I Do Wrong | 同一 writing prompt 在不同 prompt variants 下输出分布是否变化 | 衡量 prompt sensitivity |
| Originality | Haase et al. | 如果任务是创造性写作或 idea generation，可评价输出新颖性 | 衡量开放式输出质量 |
| Fluency | Haase et al. | 评价有效内容数量、表达流畅度或可用 idea 数量 | 衡量开放式输出质量 |
| Within-LLM variance | Haase et al. | 同一 prompt 多次生成，估计自然输出差异 | 校准 sample noise |

**Apple-to-apple 比较方式**

Open-ended writing 的 apple-to-apple 比较不能用 correctness，而应使用参考文献中的 output quality / variability 准则：

```text
clean prompt repeated outputs
vs.
perturbed prompt repeated outputs
```

比较对象包括：

```text
response diversity
semantic coherence
originality
fluency
sensitivity
within-LLM variance
```

Haase et al. 的关键作用是：不要把单次 open-ended 输出差异直接归因于 prompt perturbation。必须先估计同一 prompt 下的 within-LLM sampling variance。

**推荐主指标**

```text
Sensitivity + response diversity + semantic coherence + originality / fluency + within-prompt sample variance
```

如果 writing 任务不是创造力任务，而是一般开放式说明文或摘要，则 originality 可以弱化，semantic coherence 和 sensitivity 更重要。

## 四类任务的评价准则矩阵

| 任务类型 | PDR / APDR | Sensitivity | Consistency | Semantic preservation / coherence | Response diversity | Originality / Fluency | Within-LLM variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Factual QA | 主指标 | 主指标 | 可选 | 主指标 | 可选 | 不适合 | 主指标 |
| Math Reasoning | 主指标 | 主指标 | 可选 | 主指标 | 可选 | 不适合 | 主指标 |
| Code Generation | 主指标，基于 pass rate | 主指标 | 主指标 | 主指标 | 可选 | 不适合 | 主指标 |
| Open-Ended Writing | 不适合，除非有人工评分标准 | 主指标 | 可选 | 主指标 | 主指标 | 主指标，尤其创造性任务 | 主指标 |

## 如何用于 sample-noise 校准后的 apple-to-apple 比较

本项目真正要证明的不是“发明一个新评价准则”，而是：

> 使用参考文献已有评价准则时，single-generation evaluation 可能高估 prompt perturbation effect；repeated sampling 和 within-prompt noise calibration 可以更准确地估计扰动影响。

因此，每个任务都应该在同一评价准则下比较两种估计方式：

| 比较对象 | 含义 |
|---|---|
| Single-sample estimate | 每个 clean / perturbed prompt 只生成一次，然后计算 PDR、sensitivity 或 output drift |
| Repeated-sampling estimate | 每个 clean / perturbed prompt 生成多次，先估计 within-prompt variance，再判断 between-prompt difference 是否超过 sample noise |

### 对有标准答案任务

适用任务：

```text
factual QA
math reasoning
code generation
```

参考文献准则：

```text
PDR / APDR
Sensitivity
Consistency
```

比较方式：

```text
single-sample PDR
vs.
repeated-sampling PDR
```

以及：

```text
observed correctness change
vs.
correctness change after accounting for repeated-sampling variability
```

如果校准后 PDR 下降，说明单次生成把 sample noise 误当成 prompt perturbation effect。

### 对开放式生成任务

适用任务：

```text
open-ended writing
```

参考文献准则：

```text
response diversity
semantic coherence
originality
fluency
within-LLM variance
```

比较方式：

```text
single-output difference
vs.
between-prompt difference relative to within-prompt variance
```

如果 perturbed prompt 和 clean prompt 的差异没有超过同一 prompt 重复生成的自然差异，就不能强结论说 perturbation 改变了模型行为。

## 推荐写进 proposal 的版本

```text
Following prior work, this study evaluates prompt perturbation effects using task-appropriate reference metrics rather than introducing a new metric. For factual QA, mathematical reasoning, and code generation, the primary criterion follows PromptRobust and Enhancing LLM Robustness: performance degradation under perturbation, measured as Performance Drop Rate (PDR), with performance instantiated as answer correctness, final-answer accuracy, or unit-test pass rate. For black-box stability, the study adopts the sensitivity and consistency framing from What Did I Do Wrong? For open-ended writing, where no single ground-truth answer exists, the evaluation follows POSIX and Haase et al. by examining response diversity, semantic coherence, originality/fluency where applicable, and within-model sampling variance.

The methodological contribution is not a new evaluation criterion, but an apple-to-apple comparison of these reference criteria before and after sample-noise calibration. Each clean and perturbed prompt is sampled repeatedly. The study then tests whether between-prompt differences exceed the within-prompt variability emphasized by Haase et al. This allows perturbation effects measured by existing criteria such as PDR, sensitivity, coherence, or fluency to be interpreted more accurately.
```

## 最终推荐

项目中最稳妥的评价设计是：

| 任务 | 主评价准则 | 辅助评价准则 | sample-noise 校准依据 |
|---|---|---|---|
| Factual QA | PDR based on answer correctness | Sensitivity；semantic preservation | Haase et al. within-LLM variance |
| Math Reasoning | PDR based on final-answer correctness | Sensitivity；semantic preservation | Haase et al. within-LLM variance |
| Code Generation | PDR based on unit-test pass rate | Sensitivity；consistency；semantic preservation | Haase et al. within-LLM variance |
| Open-Ended Writing | Sensitivity；semantic coherence；response diversity | Originality；fluency | Haase et al. within-LLM variance |

其中，similarity 和 correctness 可以继续保留，但需要明确它们分别对应参考文献中的两个传统方向：

```text
correctness -> PDR / APDR / Micro-F1
similarity  -> sensitivity / semantic coherence / response diversity / within-LLM variance
```
