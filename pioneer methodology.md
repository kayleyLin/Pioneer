# RQ1 Methodology Notes

## Purpose Of This Document

This document records the current RQ1 methodology design for the Pioneer research project. It is written as a working methodology draft, not as final paper prose.

The main purpose is to keep the RQ1 methodology clear and reproducible. The shared parts of the experiment, such as task selection, prompt sampling, model choice, perturbation construction, and generation settings, are described first. The RQ1 analysis procedure is then documented separately.

Current scope:

```text
RQ1: Noise-corrected prompt sensitivity across task types.
RQ2: Not included in the current experiment.
RQ3: Not included in the current experiment.
```

## 1. Overall Research Design

The study investigates how changes in prompts affect LLM outputs. A central methodological issue is that LLM outputs can vary even when the exact same prompt is submitted multiple times. Therefore, the design separates two sources of variation:

```text
1. Sampling noise: ordinary output variation from repeated generation using the same prompt.
2. Perturbation effect: additional output variation after the prompt wording or structure is changed.
```

This means the project first estimates a baseline level of output variation, then uses that baseline to interpret the effect of prompt perturbations.

The overall experiment has two connected stages:

| Stage | Purpose | Main output |
|---|---|---|
| Baseline stage | Measure how much outputs vary when the same prompt is repeated | Sampling-noise baseline |
| Perturbation stage | Measure how much outputs change after prompt perturbations | Noise-corrected drift |

This structure is important because a perturbed output should not automatically be treated as meaningful drift. Some difference may simply come from normal generation randomness.

## 2. Shared Experimental Setup

The following settings are used for the current RQ1 experiment.

### 2.1 Task Types And Dataset Selection

The study uses four task types:

| Task type | Dataset | Reason for inclusion |
|---|---|---|
| factual_qa | SQuAD V2 | Context-question factual QA with a constrained answer target and reorderable prompt structure |
| math_reasoning | MATH / Hendrycks MATH | Mathematical reasoning problems aligned with PromptRobust / PromptBench's mathematics task category |
| code_generation | HumanEvalPack Python | Programming tasks where different code can be functionally equivalent |
| open_ended_writing | Alpaca | Open-ended instruction following with a broad output space |

The task categories are based on output structure and evaluation logic, not only on dataset names.

Math reasoning and code generation are separated even though both are objectively verifiable tasks. The reason is that they define correctness and output variation in different ways. In math reasoning, the main target is usually a final numeric answer, and variation often appears in the explanation path leading to that answer. Two math outputs can use different reasoning steps but still be equivalent if they reach the same final value. In code generation, however, the target is not a single textual answer. The output is an executable or function-like artifact, and correctness depends on whether the code behaves correctly under tests.

This difference matters for RQ1 because semantic similarity has different meanings across the two tasks: a math solution can remain semantically stable if the final answer and reasoning are similar, while code can be semantically or textually different because of variable names, helper functions, comments, type hints, or control-flow choices while still solving the same problem. Combining the two into one broad "verifiable task" category would therefore mix two different sources of variation and make the task-level baseline harder to interpret.

### 2.2 Prompt Sampling

The current prompt selection method is stratified random sampling by task type. In this method, each task type is treated as a separate sampling stratum. The prompts are first divided into factual QA, math reasoning, code generation, and open-ended writing groups. Then the same number of prompts is randomly selected from each group.

```text
random_seed = 20260623
sample_size = 10 prompts per task type
total_sample_size = 40 original prompts
```

Expanded formal sample prepared after the first statistical analysis:

```
random_seed = 20260623
sample_size = 50 prompts per task type
total_sample_size = 200 original prompts
sample_file = prompts/rq1_sampled_original_prompts_n50.csv
```

The expanded sample keeps the same stratified random sampling algorithm. The only methodological change is the number of prompts selected from each task-type stratum. This expansion is motivated by the completed 10-prompt-per-task formal run, where the task-level sampling-noise baseline had relatively high standard deviation and the one-way ANOVA did not find a statistically significant task-type difference. Increasing the number of prompts per task type is intended to produce more stable task-level estimates and improve statistical power.

Current expanded-sample status:

```text
Completed:
- sampled 50 prompts per task type, 200 original prompts total
- created and validated a non-API dry-run perturbation file with 1000 rows
- confirmed that all non-paraphrasing perturbations change the prompt text

Pending:
- generate the final n=50 perturbation file with GPT-3.5-Turbo paraphrasing
- generate original-prompt outputs for the n=50 sample
- generate perturbed-prompt outputs for the n=50 sample
- rerun Sentence-BERT similarity, baseline correction, heatmaps, and statistical tests
```

The algorithm can be summarized as:

```text
1. Define the task-type strata.
2. Load the dataset linked to each task type.
3. Assign each available prompt a source index.
4. Set the random seed to 20260623.
5. Randomly sample 10 prompts from each task-type stratum.
6. Save the sampled prompts and metadata.
```

Stratified sampling keeps the formal sample balanced, so one task type does not dominate the analysis. This is more appropriate than simple random sampling for the current study because the research question compares output variation across task types.

Sampling files:

```text
src/06_sample_benchmark_prompts.py
docs/rq1_sampling_algorithm.md
docs/rq1_sampling_record.md
prompts/rq1_sampled_original_prompts.csv
```

The sampled prompt file records the task type, dataset name, dataset split, source index, prompt text, reference answer when available, and random seed. This makes the formal sample reproducible.

### 2.3 Model Selection And Generation Controls

The current formal RQ1 experiment uses:

```text
model = gpt-4o-mini
provider = OpenAI API
```

The same model is used for the pilot trial and the formal RQ1 output-generation experiment. The model was selected as a practical model because it is accessible through the API, relatively low-cost, and capable across factual QA, math reasoning, code generation, and open-ended writing. The current study does not compare multiple LLMs. Therefore, model choice should be described as a controlled experimental choice rather than a final claim that this model is optimal.

Current generation settings:

```text
temperature = 0.7
top_p = 0.9
n_samples_per_prompt = 5
max_output_tokens = not set
system_prompt = "You are a helpful assistant. Answer the user's prompt directly."
```

Paraphrasing uses a separate perturbation-generation model because POSIX generates paraphrases with GPT-3.5-Turbo:

```text
paraphrase_generation_model = gpt-3.5-turbo
paraphrase_generation_method_source = Reference 3 POSIX
```

Configuration file:

```text
config/rq1_generation_config.json
```

The key control principle is that the same generation settings are used for original prompts and perturbed prompts. This keeps the comparison focused on prompt changes rather than model-parameter changes.

### 2.4 Parameter Status

Some settings are fixed pilot choices and have not yet been optimized through separate experiments.

| Setting | Current value | Status |
|---|---:|---|
| model | gpt-4o-mini | Fixed output-generation model; not compared with other LLMs in current RQ1 |
| temperature | 0.7 | Pilot/proposal setting; not separately optimized |
| top_p | 0.9 | Pilot/proposal setting; not separately optimized |
| max_output_tokens | not set | Subsequent data generation does not manually set an output-token cap |
| system prompt | helpful assistant, direct answer | Fixed for control; not separately optimized |
| original prompts per task | 10 in completed formal run; 50 in expanded run | Expanded after the first formal analysis showed high baseline variability |
| generations per prompt | 5 | Formal RQ1 setting, based on repeated-generation calibration |
| output-generation model | gpt-4o-mini | Same model as the pilot trial |
| paraphrase-generation model | gpt-3.5-turbo | Follows Reference 3 POSIX |
| RQ1 perturbation items | 10 prompts per task in completed formal run; 50 prompts per task in expanded run | One version of each perturbation type is created for every selected original prompt |

Important writing note:

```text
Earlier pilot outputs may contain a max_output_tokens metadata column from the old setup, but subsequent data generation does not manually set max_output_tokens in the API request.
```

### 2.5 Literature Alignment For Classification, Datasets, And Perturbation Methods

This study is not designed as a direct replication of one prior paper. Instead, it synthesizes component-level methods from prior work on prompt robustness, prompt sensitivity, and within-model variability. The literature alignment is therefore organized by methodological component:

```text
task classification -> closest task types in prior studies
dataset selection -> closest benchmark datasets in prior studies
perturbation taxonomy -> closest perturbation categories in prior studies
perturbation construction -> closest method for adding each perturbation
evaluation method -> closest metric or evaluation logic in prior studies
```

This framing is important because no single reference paper uses exactly the same combination of four task types, five natural perturbation types, Sentence-BERT output similarity, and repeated-sampling noise correction.

#### 2.5.0 Final Operational Decision: Perturbation Addition Methods

The final study uses one primary perturbation-addition method for each perturbation type. These methods are selected based on literature alignment, reproducibility, and task-validity control. To avoid introducing a separate weighting problem across multiple selection criteria, the study does not choose among methods through a multi-dimensional scoring system. Instead, the final operational method is fixed before the main experiment and then applied consistently across task types.

Reference key for this subsection:

| Reference number | Paper used for method alignment |
|---|---|
| Reference 1 | PromptRobust / PromptBench: prompt robustness with character-, word-, sentence-, and semantic-level perturbations |
| Reference 2 | Enhancing LLM Robustness to Perturbed Instructions |
| Reference 3 | POSIX: A Prompt Sensitivity Index for Large Language Models |
| Reference 4 | What Did I Do Wrong? Quantifying LLM Sensitivity and Consistency |
| Reference 5 | Within-Model vs Between-Prompt Variability in Large Language Models |

| Perturbation type | Final method used to add the perturbation | Specific literature method followed | Adoption status | Implementation rule |
|---|---|---|---|---|
| paraphrasing | LLM-assisted paraphrase generation followed by manual semantic-equivalence checking | Reference 3, POSIX: uses GPT-3.5-Turbo to generate paraphrases that preserve the original prompt intent and meaning. | Directly follows the POSIX LLM-paraphrase method, with manual equivalence checking added as a control step | Generate one candidate paraphrase for each original prompt; reject and regenerate if the answer target, constraints, numbers, entities, examples, or code signature change. |
| reordering | Rule-based prompt-component reordering | Reference 5, Haase et al.: uses an Information Order prompt variation, where the same information is presented in a different order. | Directly follows the information-order method | Identify prompt components such as instruction, context, question, constraints, and output requirement; move their order while keeping all information unchanged. |
| formatting_changes | Rule-based template and format transformation | Reference 3, POSIX: uses prompt-template variations that preserve prompt meaning while changing prompt structure, labels, separators, capitalization, and answer markers. Reference 5, Haase et al.: uses formatting-tweak prompt variations. | Directly follows POSIX-style template variation, with Haase et al. as secondary support | Convert prose into bullets, numbered fields, labels, or list format; add or remove polite phrases only when the request meaning stays unchanged. |
| context_injection | PromptRobust-style irrelevant sentence insertion using a fixed neutral sentence bank | Reference 1, PromptRobust: uses sentence-level attacks, including StressTest and CheckList-style irrelevant or extraneous sentence insertion/appending. | Adapts PromptRobust sentence-level insertion from adversarial distractors to natural irrelevant background context | Add one irrelevant but non-conflicting background sentence; the sentence must not provide evidence, hints, examples, assumptions, or new task conditions. |
| surface_noise | POSIX-style rule-based spelling-error injection | Reference 3, POSIX: randomly selects tokens and applies spelling-error operations, including insertion, omission, transposition, and substitution. Reference 1, PromptRobust and Reference 2, Enhancing LLM Robustness also support character-level perturbations such as TextBugger / DeepWordBug. | Directly follows POSIX spelling-error operations; PromptRobust and Enhancing LLM Robustness are secondary support | Apply insertion, omission, transposition, or substitution to non-critical instruction words only; do not alter numbers, entities, formulas, code signatures, examples, or answer-critical text. |

This decision means that the perturbation construction strategy is hybrid:

```text
paraphrasing -> Reference 3 POSIX-style LLM paraphrase generation, with manual checking
reordering -> Reference 5 Haase et al. information-order variation
formatting changes -> Reference 3 POSIX-style prompt-template variation
context injection -> Reference 1 PromptRobust-style irrelevant sentence insertion, softened for natural context
surface noise -> Reference 3 POSIX-style spelling-error operations
```

The rationale is that different perturbation types have different failure risks. Paraphrasing needs natural language flexibility, while reordering, formatting, and surface noise need reproducibility and strict control. Context injection is treated conservatively because irrelevant background information can easily become a hidden hint or a new condition if generated freely.

Operationally, the main experiment should not introduce additional perturbation-generation methods beyond this table. If a perturbed prompt fails the semantic-equivalence check, the same selected method should be reapplied to generate a replacement rather than switching to a different method.

#### 2.5.1 Task Classification And Dataset Alignment

The task categories are defined by output structure and evaluation logic, not only by dataset name. This is why math reasoning and code generation are separated even though both can be objectively evaluated: math answers may vary in reasoning steps and final numeric expression, while code outputs may vary in variable names, helper functions, style, and implementation structure while still being functionally correct.

| Project task type | Current dataset | Closest literature correspondence | Match status | Methodological reason |
|---|---|---|---|---|
| factual_qa | SQuAD V2 | PromptRobust / PromptBench reading comprehension and QA tasks, especially SQuAD V2 | Direct / high | Factual QA now uses a literature-aligned context-question dataset. The context-question structure also makes information reordering methodologically cleaner than with single-question TriviaQA prompts. |
| math_reasoning | MATH / Hendrycks MATH | PromptRobust / PromptBench mathematics task category, especially MATH / Mathematics | Direct / high | Math reasoning now uses a literature-aligned MATH dataset rather than GSM8K. MATH has objective final-answer correctness while still allowing variation in reasoning paths. |
| code_generation | HumanEvalPack Python | No exact code-generation benchmark appears in the current reference set | Project extension | Code generation is retained as an extension because it is objectively evaluable through functional correctness, but textual similarity and functional correctness can diverge. This makes it useful for comparing semantic drift with task correctness. |
| open_ended_writing | Alpaca | POSIX open-ended generation with Alpaca; Haase et al. open-ended / creative generation tasks | Direct or high match | Open-ended writing has a large output space, making semantic drift and sampling noise especially relevant. Alpaca aligns directly with POSIX's open-ended generation setting. |

Writing implication:

```text
The project can claim that factual QA, math reasoning, and open-ended writing are strongly grounded in prior prompt-robustness or prompt-sensitivity work. Code generation should be described as a deliberate extension, not as a directly replicated task from the current reference set.
```

#### 2.5.2 Perturbation Taxonomy Alignment

The five perturbation categories follow the original proposal's Perturbation Design section. Prior work is used as support for why each category is methodologically reasonable, but the final definitions are kept consistent with the proposal.

| Project perturbation type | Definition used in this study | Literature correspondence | Match status |
|---|---|---|---|
| paraphrasing | Meaning-preserving rewording | POSIX paraphrases; What Did I Do Wrong prompt rephrasings; Haase et al. phrasing changes | Strong / direct |
| reordering | Rearranging sentences or the order in which information is presented | Haase et al. information-order prompt variation; POSIX template-structure variation as secondary support | Strong / direct for information order |
| formatting_changes | Converting text to a list or adding/removing polite phrases | POSIX prompt templates; Haase et al. formatting tweaks; What Did I Do Wrong rephrasing variants | Strong / close |
| context_injection | Adding background information unrelated to the question | PromptRobust sentence-level attacks, including StressTest and CheckList-style irrelevant or extraneous sentence insertion | Adapted from more adversarial methods |
| surface_noise | Simulating unintentional spelling or punctuation errors, such as missing letters or extra spaces | POSIX spelling errors; PromptRobust character-level perturbations such as TextBugger / DeepWordBug; Enhancing LLM Robustness DeepWordBug; Haase et al. typo / random-error robustness | Strong / close, adapted to natural low-intensity noise |

Important distinction:

```text
PromptRobust and Enhancing LLM Robustness often use adversarial perturbations designed to reduce model performance.
This project uses natural, everyday perturbations. Therefore, adversarial methods from the literature are used as methodological support, but they are softened so that the perturbed prompt remains semantically equivalent to the original prompt.
```

#### 2.5.3 How Perturbations Are Added

Each perturbation changes only one aspect of the prompt at a time. Task-critical information should remain unchanged, including factual entities, numbers, math conditions, code signatures, examples, and reference-answer targets.

| Perturbation type | How the perturbation is added in this project | Literature method being followed or adapted | Control rule |
|---|---|---|---|
| paraphrasing | Rewrite the prompt with different wording while preserving the same task intent | Reference 3 POSIX uses GPT-3.5-Turbo paraphrases | The answer target and task constraints must not change. |
| reordering | Move the order of prompt components, such as instruction, context, question, constraints, or output requirement | Haase et al. information-order prompt variation | Only the order changes; no information is added, removed, or reinterpreted. |
| formatting_changes | Convert prose into bullets, numbered fields, labels, or a list; add or remove polite phrases when this does not change task meaning | POSIX prompt-template variation; Haase et al. formatting tweaks | The same semantic content remains present. |
| context_injection | Add one irrelevant but non-conflicting background sentence or phrase | PromptRobust sentence-level attacks that append irrelevant or extraneous sentences, adapted into a natural-context version | The added context must not provide evidence, hints, examples, or new conditions. |
| surface_noise | Add minor spelling, punctuation, spacing, or missing-letter errors that resemble normal user input | POSIX spelling-error operations; PromptRobust character-level attacks; Enhancing LLM Robustness DeepWordBug | Noise should be recoverable and should not corrupt numbers, formulas, names, code signatures, or answer-critical text. |

For the formal study, the final method is LLM-assisted only for paraphrasing. The other four perturbation types are constructed through rule-based, template-based, or fixed-bank procedures. This is defensible because the design prioritizes semantic equivalence, reproducibility, and isolation of the perturbation type.

#### 2.5.4 Dataset And Method Status

| Component | Current project choice | Literature status | Writing status |
|---|---|---|---|
| factual QA dataset | SQuAD V2 | Directly aligned with PromptRobust / PromptBench QA tasks | Describe as literature-aligned |
| math dataset | MATH / Hendrycks MATH | Directly aligned with PromptRobust / PromptBench mathematics tasks | Describe as literature-aligned |
| code dataset | HumanEvalPack Python | No exact match in the current six references | Describe as project extension |
| open-ended dataset | Alpaca | Directly aligned with POSIX open-ended generation | Describe as direct/high alignment |
| semantic similarity metric | Sentence-BERT cosine similarity | Directly supported by Sentence-BERT | Describe as direct metric support |
| repeated sampling / noise baseline | Multiple generations from the same prompt | Conceptually supported by Haase et al.'s within-model variability design | Describe as adapted repeated-sampling logic |

#### 2.5.5 Summary Of Literature-Grounded Design Logic

The methodology can be summarized as follows:

```text
Task types are selected to cover different output structures: constrained factual answers, numeric reasoning, executable code, and open-ended writing.
Dataset choices are mostly aligned with prior prompt-robustness benchmarks, with code generation added as an explicit extension.
Perturbation categories follow the original proposal and are supported by prior work on paraphrases, prompt templates, information order, character-level noise, and sentence-level irrelevant context.
Perturbations are added in a controlled way so that they change wording, order, format, context, or surface form without changing the underlying task.
Output similarity is measured with Sentence-BERT cosine similarity, while repeated sampling is used to estimate ordinary generation noise before interpreting perturbation effects.
```

## 3. RQ1 Methodology: Noise-Corrected Prompt Sensitivity

RQ1 asks whether different prompt perturbation types affect output semantic similarity in the same way across task types, or whether the ranking of perturbation effects is task-dependent.

The proposal-level RQ1 can be operationalized as:

```text
After applying a sampling-noise correction, is the ranking of five perturbation types by their effect on output semantic similarity consistent across factual QA, math reasoning, code generation, and open-ended writing?
```

For implementation, RQ1 is divided into two methodological parts:

| Part | Role |
|---|---|
| RQ1a | Estimate the sampling-noise baseline using repeated generations from the same original prompt |
| RQ1b | Apply perturbations and calculate noise-corrected semantic drift |

RQ1a is not a separate final research question. It is the baseline needed to interpret RQ1b.

### 3.1 RQ1a: Sampling-Noise Baseline

For each original prompt:

```text
1. Submit the exact same prompt to the same LLM multiple times.
2. Keep model, temperature, top_p, and system prompt fixed.
3. Save every generated output.
4. Compute pairwise semantic similarity among outputs from the same prompt.
5. Average the pairwise similarities.
6. Convert similarity into drift using 1 - mean_similarity.
```

For the formal setting with five outputs per prompt, the within-prompt baseline uses all pairwise comparisons among the five outputs:

```text
similarity(output_1, output_2)
similarity(output_1, output_3)
...
similarity(output_4, output_5)

number_of_pairs = 5 choose 2 = 10
mean_within_prompt_similarity = average of the 10 similarities
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

The task-level baseline is calculated by averaging prompt-level sampling-noise drift within each task type.

### 3.2 Formal RQ1a Baseline Status

Current formal RQ1 baseline design:

```text
40 original prompts
10 prompts per task type
5 generations per prompt
200 total baseline outputs
similarity metric = Sentence-BERT cosine similarity
```

Formal baseline calculation status:

The formal original-prompt output generation has been completed. The file `outputs/rq1_formal_original_generations.csv` contains 200 outputs: 40 original prompts x 5 repeated generations. The formal task-level baseline has also been calculated with Sentence-BERT.

Formal task-level baseline results:

| Task type | n items | Mean within-prompt similarity | Mean sampling-noise drift | SD sampling-noise drift |
|---|---:|---:|---:|---:|
| code_generation | 10 | 0.931653 | 0.068347 | 0.028244 |
| factual_qa | 10 | 0.951691 | 0.048309 | 0.064513 |
| math_reasoning | 10 | 0.913485 | 0.086515 | 0.044816 |
| open_ended_writing | 10 | 0.921510 | 0.078490 | 0.072555 |

Older pilot baseline results are retained below only as historical pipeline evidence:

| Task type | Mean similarity | Mean sampling-noise drift | SD drift |
|---|---:|---:|---:|
| factual_qa | 0.949619 | 0.050381 | 0.049847 |
| math_reasoning | 0.939735 | 0.060265 | 0.031659 |
| code_generation | 0.942352 | 0.057648 | 0.026224 |
| open_ended_writing | 0.909664 | 0.090336 | 0.086416 |

Interpretation:

```text
Open-ended writing shows the highest baseline drift, meaning that repeated outputs from the same prompt are naturally more variable.
Factual QA, math reasoning, and code generation show lower baseline drift in this pilot.
```

Files:

```text
outputs/rq1_formal_original_generations.csv
outputs/sbert_rq1_formal_baseline_by_item.csv
outputs/sbert_rq1_formal_baseline_by_task.csv

Historical pilot files:
outputs/rq1_generations.csv
outputs/sbert_rq1a_noise_by_item.csv
outputs/sbert_rq1a_noise_by_task.csv
rq1a_baseline_results.md
```

### 3.3 Repeated-Generation Calibration

A calibration experiment was added because the original pilot used three generations per prompt, and the project needed to check whether that number was stable enough.

Calibration design:

```text
2 prompts per task type
8 prompts total
10 generations per prompt
baseline recalculated at n = 3, 5, 7, and 10
similarity metric = Sentence-BERT cosine similarity
```

Calibration results:

| Task type | n=3 | n=5 | n=7 | n=10 |
|---|---:|---:|---:|---:|
| code_generation | 0.073214 | 0.065846 | 0.062641 | 0.058804 |
| factual_qa | 0.041099 | 0.035543 | 0.031909 | 0.031575 |
| math_reasoning | 0.077373 | 0.063475 | 0.061063 | 0.060591 |
| open_ended_writing | 0.178248 | 0.169233 | 0.165099 | 0.168513 |

Conclusion:

```text
n = 3 is acceptable for a low-cost pilot.
n = 5 is recommended for the formal RQ1 experiment if budget allows.
Increasing beyond n = 5 gives smaller additional benefit in this pilot.
```

The logic is that n = 3 gives only 3 pairwise comparisons per prompt, while n = 5 gives 10 pairwise comparisons per prompt. This makes the baseline estimate more stable without increasing API cost as much as n = 7 or n = 10.

Files:

```text
src/13_create_rq1_calibration_set.py
src/14_generate_rq1_calibration_outputs_openai.py
outputs/rq1_calibration_generations.csv
outputs/sbert_rq1_baseline_stability_by_task.csv
rq1_baseline_stability_calibration.md
```

### 3.4 RQ1b: Perturbation Type Design

RQ1b uses five perturbation types:

| Perturbation type | What changes | What should remain stable |
|---|---|---|
| paraphrasing | Meaning-preserving rewording | Core task meaning |
| reordering | Sentence order or the order in which information is presented | Required task and answer target |
| formatting_changes | Text converted to a list or polite phrases added/removed | Semantic content |
| context_injection | Background information unrelated to the question | The original question and task constraints |
| surface_noise | Unintentional spelling or punctuation errors, such as missing letters or extra spaces | Recoverable task meaning |

The reason for using these five categories is that they represent different levels of prompt change:

```text
paraphrasing -> meaning-preserving rewording
reordering -> rearranged sentence or information order
formatting_changes -> list conversion or polite-phrase changes
context_injection -> unrelated background information
surface_noise -> realistic user-input spelling or punctuation errors
```

This makes the perturbation set broader than simply rewriting a prompt in one way.

### 3.5 Perturbation Construction Procedure

Current formal RQ1 perturbation design:

```text
10 original prompts per task type
40 original prompts total
5 perturbation types per original prompt
200 perturbed prompts total
5 generations per perturbed prompt
1000 total perturbed-prompt outputs
```

The formal perturbed prompt file has been generated for the current 40-prompt sample. The file contains 200 perturbed prompts: 40 paraphrasing prompts, 40 reordering prompts, 40 formatting-change prompts, 40 context-injection prompts, and 40 surface-noise prompts.

The perturbations are not all produced in the same way. Paraphrasing is generated with GPT-3.5-Turbo following POSIX. The other four perturbation types are produced with rule-based, template-based, or fixed-bank procedures so that the change is reproducible and controlled.

Formal RQ1 perturbation construction is fixed as:

```text
paraphrasing -> POSIX-style GPT-3.5-Turbo paraphrase generation, followed by manual checking
reordering -> Haase et al. information-order variation, implemented through rule-based component reordering
formatting_changes -> POSIX-style prompt-template transformation
context_injection -> PromptRobust-style irrelevant sentence insertion, softened into a fixed neutral sentence bank
surface_noise -> POSIX-style spelling-error operations
```

The formal RQ1 setting uses 5 repeated output generations per original prompt and 5 repeated output generations per perturbed prompt. Every perturbed prompt remains marked as pending until manual semantic-equivalence checking confirms that it does not change the task meaning.

The dry-run file is only a pipeline check. It confirms that the perturbation-generation script writes the expected 40 prompts x 5 perturbation types structure. In the dry-run file, paraphrasing rows are intentionally unchanged because the dry-run does not call the OpenAI API. The full formal file uses API-generated paraphrases. The current dry-run check found zero unchanged non-paraphrasing perturbations.

Files:

```text
prompts/rq1_formal_perturbed_prompts.csv
prompts/rq1_formal_perturbed_prompts_dry_run.csv
prompts/archive/rq1_formal_perturbed_prompts_obsolete_trivia_gsm8k.csv
src/19_create_rq1_perturbed_prompts.py
src/11_generate_rq1b_perturbed_outputs_openai.py
outputs/rq1_formal_perturbed_generations.csv
```

### 3.6 RQ1b: Noise-Corrected Drift Calculation

RQ1b compares outputs from original prompts with outputs from perturbed prompts.

For each original prompt and perturbation:

```text
baseline_similarity = average similarity among repeated outputs from the original prompt
perturbation_similarity = average similarity between original-prompt outputs and perturbed-prompt outputs
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

Interpretation:

```text
positive value = perturbed outputs are less similar to original outputs than the original outputs are to each other
near zero = perturbation effect is close to ordinary sampling noise
negative value = perturbed outputs are at least as similar as repeated original outputs in this run
```

This correction is the core of RQ1 because it prevents the analysis from over-interpreting ordinary generation randomness.

### 3.7 Formal RQ1b Results

Formal noise-corrected drift heatmap:

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.034560 | 0.112586 | 0.005822 | 0.175192 |
| reordering | 0.007690 | 0.041203 | -0.000518 | 0.035788 |
| formatting_changes | 0.004808 | 0.026844 | -0.001141 | 0.002976 |
| context_injection | 0.002075 | -0.005502 | 0.000390 | 0.018683 |
| surface_noise | 0.001728 | -0.006236 | -0.006036 | 0.002013 |

Formal interpretation:

```text
Paraphrasing produced the largest positive noise-corrected drift for open-ended writing, factual QA, and code generation.
Math reasoning showed near-zero noise-corrected drift across all perturbation types.
Surface noise and context injection generally produced small or negative drift values.
The perturbation ranking is therefore task-dependent rather than uniform across task types.
```

### 3.8 Historical RQ1b Pilot Results

The following values come from an earlier small RQ1b pilot and should not be reported as the formal RQ1 result. They are retained only to document that the analysis pipeline was tested before the formal 40-prompt run.

| Perturbation type | code_generation | factual_qa | math_reasoning | open_ended_writing |
|---|---:|---:|---:|---:|
| paraphrasing | 0.041442 | -0.008114 | -0.005167 | -0.004696 |
| reordering | 0.014932 | -0.020535 | 0.024498 | 0.004064 |
| formatting_changes | -0.009633 | 0.050575 | 0.066452 | 0.000425 |
| context_injection | 0.025948 | 0.062831 | 0.005024 | 0.066072 |
| surface_noise | -0.000620 | -0.010871 | -0.012020 | 0.006697 |

Pilot interpretation:

```text
The strongest perturbation type appears task-dependent in the pilot.
Formatting changes are relatively stronger for math reasoning.
Context injection is relatively stronger for factual QA and open-ended writing.
Paraphrasing has a clearer effect for code generation in this small pilot.
```

Important limitation:

```text
These pilot results used a much smaller perturbation sample than the current formal design.
The formal RQ1 perturbation file now contains 40 original prompts x 5 perturbation types = 200 perturbed prompts.
```

Files:

```text
outputs/sbert_rq1b_perturbation_effects_by_item.csv
outputs/sbert_rq1b_perturbation_summary.csv
outputs/sbert_rq1b_heatmap_noise_corrected_drift.csv
rq1b_pilot_perturbation_results.md
```

## 4. Similarity Measurement

The current analysis uses Sentence-BERT rather than the earlier bag-of-words similarity metric.

Current model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Mechanism:

```text
1. Convert each output text into a dense semantic embedding.
2. Compute cosine similarity between embeddings.
3. Use the cosine score as the semantic similarity between two outputs.
```

Sentence-BERT is more appropriate than bag-of-words overlap because it can capture semantic similarity even when two outputs use different wording. This is especially important for paraphrasing, open-ended writing, and code explanations.

The earlier bag-of-words results should be treated only as an initial pipeline check, not as the main result.

Analysis script:

```text
src/16_analyze_rq1_with_sentence_bert.py
src/22_analyze_rq1_formal_baseline_sbert.py
```

## 5. Experimental Environment And Files

Python environment:

```text
Anaconda Python: /opt/anaconda3/bin/python
requirements.txt
datasets
openai
certifi
sentence-transformers
torch
transformers
```

Main scripts:

```text
src/06_sample_benchmark_prompts.py
src/07_generate_rq1_outputs_openai.py
src/11_generate_rq1b_perturbed_outputs_openai.py
src/13_create_rq1_calibration_set.py
src/14_generate_rq1_calibration_outputs_openai.py
src/16_analyze_rq1_with_sentence_bert.py
src/19_create_rq1_perturbed_prompts.py
```

Main prompt files:

```text
prompts/rq1_sampled_original_prompts.csv
prompts/rq1_formal_perturbed_prompts.csv
prompts/rq1_formal_perturbed_prompts_dry_run.csv
prompts/rq1_sampled_original_prompts_n50.csv
prompts/rq1_formal_perturbed_prompts_n50_dry_run.csv
prompts/rq1_formal_perturbed_prompts_n50.csv
prompts/rq1_calibration_prompts.csv
prompts/archive/rq1_formal_perturbed_prompts_obsolete_trivia_gsm8k.csv
```

Main output/result files:

```text
outputs/rq1_formal_original_generations.csv
outputs/rq1_formal_perturbed_generations.csv
outputs/sbert_rq1_formal_baseline_by_item.csv
outputs/sbert_rq1_formal_baseline_by_task.csv
outputs/sbert_rq1_formal_perturbation_effects_by_item.csv
outputs/sbert_rq1_formal_perturbation_summary.csv
outputs/sbert_rq1_formal_heatmap_noise_corrected_drift.csv
outputs/rq1_calibration_generations.csv
Historical pilot files:
outputs/rq1_generations.csv
outputs/rq1b_pilot_perturbed_generations.csv
outputs/sbert_rq1a_noise_by_task.csv
outputs/sbert_rq1b_heatmap_noise_corrected_drift.csv
outputs/sbert_rq1_baseline_stability_by_task.csv
```

Written result notes:

```text
rq1a_baseline_results.md
rq1b_pilot_perturbation_results.md
rq1_baseline_stability_calibration.md
rq1_results.md
session7_progress_notes.md
```

## 6. Current Limitations And Next Steps

Current limitations:

```text
1. The completed formal RQ1 run uses 10 prompts per task type; an expanded 50-prompt-per-task run has been prepared to address sample-size and statistical-power limitations.
2. RQ1 uses one output-generation LLM.
3. temperature and top_p are fixed pilot settings, not optimized settings.
4. The current methodology is limited to RQ1 and does not include correctness-change analysis.
```

Recommended next steps:

```text
1. Optionally run PDR / correctness-based evaluation for factual QA and math reasoning.
2. Decide whether code-generation correctness evaluation is feasible within the remaining time.
3. Convert the formal RQ1 baseline and perturbation results into paper-style Results prose.
4. Add limitations around sample size, single-model design, and SBERT-only semantic similarity.
```
