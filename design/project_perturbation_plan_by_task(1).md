# 项目适用的 Prompt Perturbation 方法总结

项目目标是评估 LLM 输出在自然 prompt 扰动下的稳定性，并区分真正的 perturbation effect 与 repeated generation 造成的 sample noise。项目包含四类任务：

1. Factual question answering
2. Mathematical reasoning
3. Code generation
4. Open-ended writing / open-ended generation

因此，本项目不应直接采用纯 adversarial attack 方案作为主实验方法。更合适的设计是：使用自然、语义保持或近似语义保持的 prompt perturbations，并对每个原始 prompt 和扰动 prompt 都进行 repeated sampling。

## 推荐的总体扰动类型

| 扰动类型 | 是否推荐作为主实验 | 适用任务 | 参考来源 | 说明 |
|---|---|---|---|---|
| Paraphrasing / rewording | 推荐 | 四类任务均适用 | POSIX；What Did I Do Wrong；Haase et al. | 最核心的自然 prompt 扰动。保持任务意图不变，只改变表达方式 |
| Information reordering | 推荐 | 四类任务均适用，尤其适合 QA、math、code | Haase et al.；POSIX template changes | 改变 prompt 中要求、上下文、格式约束的呈现顺序 |
| Formatting changes | 推荐 | 四类任务均适用，尤其适合 code 和 open-ended writing | Haase et al.；What Did I Do Wrong | 改变 bullet points、numbered list、JSON-like structure、plain text 等格式 |
| Surface noise: typos / punctuation / capitalization | 推荐，但强度应低 | 四类任务均适用 | PromptRobust；POSIX；Enhancing LLM Robustness；Haase et al. | 模拟用户输入错误。建议只扰动 instruction，不扰动 task sample 或 ground-truth-critical content |
| Context injection / irrelevant sentence insertion | 谨慎推荐 | QA、writing 较适合；math、code 需谨慎 | PromptRobust sentence-level；CheckList / StressTest | 在 prompt 中加入无关但不冲突的句子，测试模型是否被干扰 |
| Instruction-level perturbation | 推荐 | 四类任务均适用 | Enhancing LLM Robustness | 固定 task sample 不变，只改写或轻微扰动 instruction template |
| Word-level synonym replacement | 谨慎使用 | writing、QA 较适合；math、code 不建议自动替换关键术语 | PromptRobust；TextFooler | 容易改变专业术语、数学条件或代码语义，需要人工筛查 |
| Adversarial greedy search | 不建议作为主实验 | 可作为扩展实验 | PromptRobust；Enhancing LLM Robustness | 与项目“自然扰动 + noise correction”的定位不完全一致 |

## 主实验建议采用的 5 类扰动

建议将项目的 perturbation taxonomy 固定为以下 5 类。它们既能覆盖五篇文献中的主要方法，又适合四类任务统一比较。

| 编号 | 项目扰动类型 | 操作定义 | 主要衡量内容 |
|---|---|---|---|
| P1 | Paraphrase | 保持任务意图不变，重新表述 instruction | 模型是否对自然改写敏感 |
| P2 | Format Change | 改变 prompt 格式，如段落改 bullet list、添加编号、要求输出格式变化 | 模型是否受格式表达影响 |
| P3 | Information Reordering | 调换 instruction、constraints、context、question 的顺序 | 模型是否依赖信息顺序 |
| P4 | Surface Noise | 在 instruction 中加入少量 typo、标点错误、大小写变化 | 模型是否对用户输入错误鲁棒 |
| P5 | Context Injection | 加入一条无关但不矛盾的句子或轻微干扰性说明 | 模型是否容易被无关上下文干扰 |

这 5 类扰动中，P1-P4 应作为主实验扰动。P5 建议作为次要或 stress-test 类型扰动，因为 context injection 更容易引入额外语义，必须保证插入内容不改变任务答案。

## 四类任务对应的推荐扰动方案

### 1. Factual QA

**任务特点**

Factual QA 关注答案是否事实正确。当前 RQ1 使用 SQuAD V2，因为它更直接对应 PromptRobust / PromptBench 中的 reading-comprehension / QA 任务。prompt 通常包含 context、question 和 answer instruction。

**推荐扰动**

| 扰动类型 | 推荐程度 | 示例操作 | 注意事项 |
|---|---|---|---|
| Paraphrase | 高 | 将 "Answer the question based on the passage" 改为 "Using only the given passage, provide the answer" | 不改变 evidence 范围 |
| Information Reordering | 高 | 将 question 放在 context 前或后；将 answer instruction 从开头移到末尾 | 不改变 context 原文 |
| Format Change | 高 | plain paragraph 改为 `Context: ... Question: ... Answer:` 格式 | 保持字段含义清晰 |
| Surface Noise | 中 | 在 instruction 中加入少量拼写错误，例如 "questoin"、"answr" | 不扰动 passage 和 question 中的实体名、数字、日期 |
| Context Injection | 中 | 加入 "Ignore unrelated information and answer from the passage only" 或一条无关背景句 | 插入内容不能提供新答案，也不能和 passage 冲突 |

**不建议**

不建议自动对 context 或 question 做 synonym replacement，因为 factual QA 对实体、日期、地点、数值非常敏感。替换一个词可能导致答案条件变化。

**推荐最终设置**

Factual QA 主实验使用：

```text
P1 paraphrase
P2 format change
P3 information reordering
P4 surface noise on instruction only
```

Context injection 可作为额外 stress test。

### 2. Mathematical Reasoning

**任务特点**

Math reasoning 对条件、数字、变量、比较关系非常敏感。任何对题目本身的扰动都可能改变正确答案。因此，扰动应主要作用于 instruction template，而不是 problem statement。

**推荐扰动**

| 扰动类型 | 推荐程度 | 示例操作 | 注意事项 |
|---|---|---|---|
| Paraphrase | 高 | 将 "Solve the problem step by step and give the final answer" 改为 "Work through the solution and state the final answer clearly" | 不改变题目条件 |
| Format Change | 高 | 要求输出 `Reasoning:` 和 `Final answer:` 两个字段 | 格式变化不能暗示不同解法 |
| Information Reordering | 中高 | 将 final-answer requirement 从末尾移到开头 | 不移动或改写 problem text 内部条件 |
| Surface Noise | 中 | 只在 instruction 中加入轻微 typo | 不能扰动数字、公式、单位、变量名 |
| Context Injection | 低 | 加入无关句子，例如 "Be careful with arithmetic" | 容易改变模型推理策略，作为 stress test 而非主扰动 |

**不建议**

不建议对 math problem 本身做 word-level synonym replacement、character-level typo 或 sentence insertion。数学题中的一个词、符号或数字变化都可能改变 ground truth。

**推荐最终设置**

Math reasoning 主实验使用：

```text
P1 paraphrase of instruction
P2 output-format change
P3 instruction-order change
P4 surface noise on instruction only
```

Context injection 建议只作为补充，并且插入句必须是通用提醒，而不是新条件。

### 3. Code Generation

**任务特点**

Code generation 的输出具有可执行性和结构约束。任务正确性通常由 unit tests 或 pass/fail 判断。prompt 扰动不应改变 function signature、input/output specification 或 hidden requirements。

**推荐扰动**

| 扰动类型 | 推荐程度 | 示例操作 | 注意事项 |
|---|---|---|---|
| Paraphrase | 高 | 将 "Write a Python function..." 改为 "Implement a Python function..." | 不改变函数名、参数、返回值要求 |
| Format Change | 高 | 将 requirement 从段落改为 bullet list；或将 examples 放入 fenced code block | 不改变 examples 内容 |
| Information Reordering | 高 | 调换 constraints、examples、function signature 的顺序 | function signature 最好保持原样 |
| Surface Noise | 中 | 在自然语言 instruction 中加入 typo | 不扰动代码片段、函数名、变量名、测试样例 |
| Context Injection | 中 | 加入 "Do not include explanations, only code" 或无关开发背景 | 不能引入新的性能、库使用或风格要求 |

**不建议**

不建议对代码签名、示例输入输出、变量名或边界条件做字符级扰动。对 code task 来说，这类扰动不再是 prompt perturbation，而可能变成 task mutation。

**推荐最终设置**

Code generation 主实验使用：

```text
P1 paraphrase
P2 format change
P3 information reordering
P4 surface noise on natural-language instruction only
P5 optional context injection
```

Code generation 是最适合检验“semantic drift 与 correctness change 是否相关”的任务之一，因为可以同时计算 output drift 和 unit-test pass rate。

### 4. Open-Ended Writing / Open-Ended Generation

**任务特点**

Open-ended writing 没有唯一标准答案，输出天然具有较高 sampling variance。因此，这类任务最需要 repeated sampling 和 noise correction。

**推荐扰动**

| 扰动类型 | 推荐程度 | 示例操作 | 注意事项 |
|---|---|---|---|
| Paraphrase | 高 | 将写作要求换一种说法 | 保持写作目的、受众、体裁不变 |
| Format Change | 高 | 要求输出段落、bullet list、短文、标题加正文等 | 格式本身会强烈影响输出，应单独作为扰动类型 |
| Information Reordering | 高 | 调换 audience、topic、tone、length constraint 的顺序 | 不改变关键写作约束 |
| Surface Noise | 中 | 加入少量 typo 或 punctuation noise | 不应影响主题词 |
| Context Injection | 高 | 加入一句无关背景或弱干扰信息 | 可测试开放式生成是否容易跑题 |

**可选扩展**

参考 Haase et al.，open-ended writing 还可以加入 prompt strategy 类型：

| Strategy | 是否建议 |
|---|---|
| Persona prompt | 可作为扩展，不建议混入主扰动 |
| One-shot example | 可作为扩展，但会显著改变 prompt information |
| Heuristic prompt | 可作为扩展，会改变输出质量和创造性 |
| Zero-shot CoT | 不建议作为普通扰动，属于 prompting strategy |

**推荐最终设置**

Open-ended writing 主实验使用全部 5 类扰动：

```text
P1 paraphrase
P2 format change
P3 information reordering
P4 surface noise
P5 context injection
```

这类任务必须重复采样。单次输出差异不能被直接解释为 perturbation effect。

## 推荐的项目主实验矩阵

| 任务类型 | Paraphrase | Format Change | Information Reordering | Surface Noise | Context Injection |
|---|---:|---:|---:|---:|---:|
| Factual QA | 主实验 | 主实验 | 主实验 | 主实验，instruction only | 可选 stress test |
| Math Reasoning | 主实验，instruction only | 主实验 | 主实验，instruction only | 主实验，instruction only | 不推荐或仅 stress test |
| Code Generation | 主实验 | 主实验 | 主实验 | 主实验，natural language only | 可选 stress test |
| Open-Ended Writing | 主实验 | 主实验 | 主实验 | 主实验 | 主实验 |

## 扰动设计原则

为了保证实验结果可解释，建议所有任务遵守以下原则：

1. **只扰动 prompt instruction，尽量不扰动 task sample。**
   对 QA、math、code 尤其重要，因为 sample 本身包含事实、数字、函数签名或测试条件。

2. **扰动后必须保持 task intent 不变。**
   如果扰动改变了正确答案，测到的就不是 prompt sensitivity，而是任务变了。

3. **每个扰动类型单独评估。**
   不要在同一个 perturbed prompt 里同时加入 paraphrase、typo、format change 和 context injection，否则无法判断是哪种扰动导致输出变化。

4. **每个 prompt condition 都要 repeated sampling。**
   对 clean prompt 和每个 perturbed prompt 都生成多次输出，用 within-prompt variability 建立 sample-noise baseline。

5. **同时报告 raw drift 和 noise-corrected drift。**
   项目核心贡献是区分 observed perturbation drift 与 ordinary sample noise。

6. **对有标准答案的任务，必须同时报告 correctness。**
   Factual QA、math、code 不能只看 embedding drift。应同时报告 answer correctness、final-answer match 或 unit-test pass rate。

7. **对开放式任务，重点报告 semantic drift 与 output-quality drift。**
   Open-ended writing 没有唯一答案，因此 semantic similarity、style/length变化、diversity 和 repeated-sampling noise 更重要。

## 最推荐的最终方案

如果需要一个简洁、可写进 proposal 的版本，可以这样定义：

> This project evaluates five natural prompt perturbation types: paraphrasing, formatting changes, information reordering, surface noise, and context injection. These perturbations are applied primarily to the instruction part of the prompt while keeping the task sample unchanged. For factual QA, mathematical reasoning, and code generation, perturbations avoid changing facts, numbers, formulas, function signatures, examples, or ground-truth-defining content. For open-ended writing, all five perturbations are used because there is no single ground-truth answer and output variability is part of the evaluation target. Each clean and perturbed prompt is sampled repeatedly, allowing raw perturbation drift to be corrected by a within-prompt sample-noise baseline.

## 与五篇参考文献的对应关系

| 项目方法 | 对应参考文献 | 项目中的改造 |
|---|---|---|
| Surface noise | PromptRobust；POSIX；Enhancing LLM Robustness；Haase et al. | 从 adversarial typo 改为轻量自然 typo，且主要作用于 instruction |
| Paraphrasing | POSIX；What Did I Do Wrong；Haase et al. | 作为主实验的自然语义保持扰动 |
| Formatting changes | Haase et al.；What Did I Do Wrong；POSIX template changes | 用于测试格式表达是否影响输出 |
| Information reordering | Haase et al.；POSIX template changes | 用于测试信息顺序敏感性 |
| Context injection | PromptRobust sentence-level perturbation | 从强 adversarial 干扰改为无关但不矛盾的自然插入 |
| Repeated sampling | Haase et al. | 转化为项目的 sample-noise baseline correction |
