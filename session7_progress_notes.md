# Session 7 Progress Notes

## July 1 Update: RQ1 Formal-Pipeline Preparation

Today the project scope was clarified again:

```text
The current experiment only includes RQ1.
RQ2 and RQ3 are not part of the current data-running plan.
```

This matters because the current goal is not correctness evaluation. The current goal is to measure noise-corrected prompt sensitivity across task types using output semantic similarity.

### 1. Formal RQ1 Settings Fixed

The formal RQ1 run now keeps the same generation settings as the pilot, except that the repeated-generation count is increased to 5.

Current configuration file:

```text
config/rq1_generation_config.json
```

Formal RQ1 settings:

| Setting | Value |
|---|---|
| active research question | RQ1 only |
| output-generation model | gpt-4o-mini |
| output-generation provider | OpenAI API |
| temperature | 0.7 |
| top_p | 0.9 |
| repeated generations per prompt | 5 |
| max_output_tokens | not set |
| paraphrase-generation model | gpt-3.5-turbo |
| paraphrase method source | Reference 3 POSIX |

Important note:

```text
The output-generation LLM is the same as the pilot trial: gpt-4o-mini.
The paraphrase-generation LLM follows POSIX: gpt-3.5-turbo.
```

### 2. Python Environment Prepared

The required Python packages were installed and verified in the Anaconda Python environment.

Environment:

```text
/opt/anaconda3/bin/python
```

Dependency record:

```text
requirements.txt
```

Verified packages include:

```text
openai
datasets
sentence-transformers
torch
transformers
certifi
httpx
httpcore
pandas
numpy
scikit-learn
```

The import smoke test passed, so the local environment can run the RQ1 generation and Sentence-BERT analysis pipeline.

### 3. Final Perturbation-Addition Methods Fixed

The five perturbation types still follow the original proposal definitions. The final operational method for adding each perturbation is:

| Perturbation type | Final addition method | Main literature source |
|---|---|---|
| paraphrasing | GPT-3.5-Turbo paraphrase generation, followed by manual semantic-equivalence checking | Reference 3 POSIX |
| reordering | Rule-based information-order change | Reference 5 Haase et al. |
| formatting_changes | Rule-based prompt-template / format transformation | Reference 3 POSIX |
| context_injection | Fixed neutral sentence bank, adapted from irrelevant sentence insertion | Reference 1 PromptRobust |
| surface_noise | POSIX-style spelling-error operations | Reference 3 POSIX |

New perturbation-generation script:

```text
src/19_create_rq1_perturbed_prompts.py
```

### 4. Formal Perturbed Prompt File Generated

The formal RQ1 perturbed prompt file was generated:

```text
prompts/rq1_formal_perturbed_prompts.csv
```

It contains:

```text
40 original prompts x 5 perturbation types = 200 perturbed prompts
```

The paraphrasing rows were checked:

```text
20 paraphrasing rows
20 rows changed from the original prompt
method_reference = Reference 3 POSIX: GPT-3.5-Turbo paraphrase generation
construction_method = llm_assisted_posix_paraphrase
```

This confirms that the paraphrasing step was generated using the configured POSIX-style GPT-3.5-Turbo paraphrase method.

### 5. Reordering Rows Checked And Manually Corrected

After generating the formal perturbed prompt file, an automatic check found 8 rows where the `reordering` perturbation was identical to the original prompt.

Reason:

```text
Some prompts were single-sentence prompts, so the rule-based reordering script did not find multiple prompt components to reorder.
```

These 8 rows were manually edited in:

```text
prompts/rq1_formal_perturbed_prompts.csv
```

After correction:

```text
unchanged rows = 0
```

The corrected reordering rows were:

| item_id | Corrected reordering |
|---|---|
| factual_qa_378 | More Christmas number one singles than any other act have been had by which act? |
| factual_qa_642 | In the film 'Mad Max Beyond the Thunderdome', who played 'Aunty Entity'? |
| factual_qa_9317 | Within the human body, the most gold is contained in which part? |
| factual_qa_12706 | Which British monarch was Mrs Maria Fitzherbert the wife of? |
| factual_qa_12920 | Along with the theatrical movies "The X-Files" and "I Want To Believe", which spin-off TV series was spawned by "The X-Files"? |
| math_reasoning_7207 | How many years in the future will it be 5 years before the 200th anniversary of the first skyscraper, if that skyscraper was built 100 years ago? |
| open_ended_writing_13494 | Describe, in a few sentences, the importance of sleep. |
| open_ended_writing_30103 | For a bedroom, name three furniture items that would be suitable. |

### 6. New Methodological Issue: Automatic Reordering Validation

A new concern was identified:

```text
Sentence-BERT can check whether two prompts are semantically similar, but it cannot check whether reordering actually occurred.
```

Therefore, a possible improvement is to add an automatic reordering-validity check before running the full experiment.

Proposed validation logic:

| Check | Method | Purpose |
|---|---|---|
| semantic preservation | Sentence-BERT cosine similarity | Confirm that the perturbation did not change the prompt meaning |
| actual order change | Kendall tau / order-distance score | Confirm that information order changed |
| prompt eligibility | at least two reorderable information units | Avoid forcing reordering onto prompts that cannot meaningfully be reordered |

Potential rule:

```text
A reordering perturbation is accepted only if semantic similarity remains high and the order-distance score indicates a non-trivial order change.
If a prompt does not contain enough reorderable units, the entire original prompt should be replaced before generating the five perturbations.
```

### 7. Literature Found For Reordering Validation

These references were found today but have not yet been read by the researcher. They should be reviewed tomorrow before deciding whether to add the automatic validation step.

| Candidate reference | Why it may help |
|---|---|
| Lapata (2006), "Automatic Evaluation of Information Ordering: Kendall's Tau" | Main candidate. Directly uses Kendall's tau for evaluating information ordering. |
| Kendall (1938), "A New Measure of Rank Correlation" | Original statistical source for Kendall tau rank correlation. |
| Kumar and Vassilvitskii (2010), "Generalized Distances between Rankings" | Useful supplement for ranking-distance metrics such as Kendall tau and Spearman footrule. |
| Li et al. (2006), "Sentence Similarity Based on Semantic Nets and Corpus Statistics" | Secondary support because it considers word order in sentence similarity, but it is less directly aligned than Lapata (2006). |

Recommended reading priority:

```text
1. Lapata (2006)
2. Kendall (1938), only for method background
3. Kumar and Vassilvitskii (2010), only if a broader ranking-distance citation is needed
4. Li et al. (2006), optional supporting reference
```

### 8. Recommended Next Steps

Tomorrow's recommended sequence:

```text
1. Read Lapata (2006) and decide whether to add Kendall tau order-distance validation.
2. Decide whether the 8 manually corrected reordering prompts are acceptable or should be replaced by newly sampled prompts.
3. If automatic validation is adopted, implement a reordering-validity check before running outputs.
4. Manually review prompts/rq1_formal_perturbed_prompts.csv for semantic equivalence.
5. Decide whether to preserve old pilot outputs or write formal outputs to new files.
6. Run original-prompt outputs with n = 5.
7. Run perturbed-prompt outputs using prompts/rq1_formal_perturbed_prompts.csv.
8. Run Sentence-BERT RQ1 analysis and update the results documents.
```

Important caution before running formal outputs:

```text
outputs/rq1_generations.csv currently contains earlier pilot outputs.
Before running the formal n = 5 experiment, decide whether to overwrite it or write formal outputs to a new file.
```

## Current Research Focus

The current focus is RQ1. RQ3 is postponed and will only be considered as an optional extension if time permits.

Updated RQ1:

```text
How much sampling noise occurs when the same prompt is repeatedly given to the same LLM, and does this sampling-noise baseline differ across task types?
```

The reason for starting with RQ1 is that prompt perturbation effects cannot be interpreted clearly unless the normal sampling variability of the model is first measured.

## What Has Been Completed

### 1. Task Type Classification

I defined four task types based on expected output structure and evaluation style:

| Task type | Dataset | Reason |
|---|---|---|
| factual_qa | SQuAD V2 | Context-question factual answers with reference answers; directly aligned with PromptRobust / PromptBench QA-style evaluation |
| math_reasoning | MATH / Hendrycks MATH | Mathematical reasoning tasks with objective final answers; closer to the mathematics datasets used in the prompt-robustness literature than GSM8K |
| code_generation | HumanEvalPack Python | Code can vary in form while still solving the same function |
| open_ended_writing | Alpaca | Open-ended responses with large output space |

Math reasoning and code generation are separated because code outputs may vary in variable names, structure, comments, helper functions, or edge-case handling, even when the implementation is functionally equivalent.

### 2. Random Sampling

Sampling method:

```text
stratified random sampling by task type
```

Pilot settings:

```text
random_seed = 20260623
sample_size_per_task_type = 10
total_sampled_prompts = 40
```

The fixed random seed makes the sampling process reproducible.

Sampled prompt file:

```text
prompts/rq1_sampled_original_prompts.csv
```

### 3. LLM Generation

For the pilot, I used one model and fixed decoding parameters.

Generation settings:

```text
model = gpt-4o-mini
n_samples_per_prompt = 3
temperature = 0.7
top_p = 0.9
max_output_tokens = 300
system_prompt = You are a helpful assistant. Answer the user's prompt directly.
```

Total generations:

```text
20 prompts x 3 outputs per prompt = 60 outputs
```

Generated output file:

```text
outputs/rq1_generations.csv
```

### 4. Similarity Analysis

For each prompt, I compared the three generated outputs pairwise:

```text
similarity(output_1, output_2)
similarity(output_1, output_3)
similarity(output_2, output_3)
```

Then I calculated:

```text
mean_within_prompt_similarity = average pairwise similarity
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

Current analysis script:

```text
src/09_analyze_rq1_real_generations.py
```

Result files:

```text
outputs/rq1_real_noise_by_item.csv
outputs/rq1_real_noise_by_task.csv
```

## Preliminary RQ1 Results

Task-level pilot results:

| Task type | Mean within-prompt similarity | Mean sampling-noise drift |
|---|---:|---:|
| factual_qa | 0.949619 | 0.050381 |
| math_reasoning | 0.939735 | 0.060265 |
| code_generation | 0.942352 | 0.057648 |
| open_ended_writing | 0.909664 | 0.090336 |

Preliminary pattern:

```text
open_ended_writing shows the highest sampling-noise drift.
factual_qa, code_generation, and math_reasoning show lower drift.
```

Possible interpretation:

```text
Tasks with larger output spaces appear to produce more variation across repeated generations. Open-ended writing has the largest possible response space, while factual QA and math reasoning have more constrained expected outputs. Code generation falls between these cases because the same function can be implemented in multiple valid ways.
```

## Important Limitations

This is a pilot result, not a final result.

Current limitations:

```text
Formal run uses 10 prompts per task type; this is larger than the pilot but still modest.
Formal run uses 5 generations per prompt.
Only one model.
The current reported metric is Sentence-BERT cosine similarity using sentence-transformers/all-MiniLM-L6-v2.
```

Planned improvement:

```text
Increase sample size after confirming the methodology.
Increase repeated generations per prompt if resources allow.
```

## Questions To Ask The Instructor

1. Is it reasonable to define RQ1 as measuring sampling noise before introducing prompt perturbations?

2. Are the four task types appropriate: factual QA, mathematical reasoning, code generation, and open-ended writing?

3. For the formal RQ1 experiment, what sample size per task type would be appropriate: 10, 15, or 20 prompts?

4. Is 5 repeated generations per prompt enough for the formal experiment, or should I use more?

5. Should Sentence-BERT be the main similarity metric, or should I include another metric as a robustness check?

6. Should code generation be evaluated only by textual similarity for RQ1, or should functional correctness be introduced later in RQ2?

## Short Verbal Update For The Meeting

```text
I revised RQ1 into two connected parts: first estimating the sampling-noise baseline, then comparing perturbation effects against that baseline. I sampled 20 prompts across four task types using stratified random sampling with a fixed seed. Then I generated three outputs for each prompt using the same model and fixed decoding parameters. I computed pairwise Sentence-BERT similarity among the repeated outputs to estimate within-prompt stability. The preliminary pilot result suggests that open-ended writing has the highest sampling-noise drift, while factual QA, math reasoning, and code generation have lower drift. These results are only preliminary because the pilot uses a small sample size, but the full RQ1 pipeline is now working.
```

## RQ1b Pilot Extension

After completing the RQ1a sampling-noise baseline, I also began a small RQ1b pilot for the perturbation part of the full RQ1.

Pilot scope:

```text
10 original prompts per task type
5 perturbation types per original prompt
200 perturbed prompts total
3 outputs per perturbed prompt
60 perturbed outputs total
```

The perturbations were manually constructed for this pilot and checked for semantic equivalence.

Current RQ1b result files:

```text
rq1b_pilot_perturbation_results.md
outputs/rq1b_pilot_heatmap_noise_corrected_drift.csv
outputs/rq1b_pilot_perturbation_summary.csv
outputs/sbert_rq1b_heatmap_noise_corrected_drift.csv
outputs/sbert_rq1b_perturbation_summary.csv
```

Main caution:

```text
This is only a pilot of the perturbation analysis because it uses one item per task type. The goal is to verify that the noise-corrected perturbation analysis pipeline works.
```
