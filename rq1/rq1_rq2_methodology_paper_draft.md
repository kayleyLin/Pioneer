# Methodology Draft For RQ1 And RQ2

## 1. Methodology Overview

This study examines the robustness of large language model (LLM) outputs under repeated generation and prompt perturbation. The methodology is designed around two related but distinct sources of output variation. First, even when the same prompt is submitted repeatedly to the same model under the same generation settings, the model may produce different outputs because of stochastic decoding. This variation is referred to in this study as sampling noise. Second, when the prompt is altered through paraphrasing, reordering, formatting changes, context injection, or surface-level noise, the output may change because of the perturbation itself. The study therefore separates baseline sampling variation from perturbation-induced semantic drift.

The current pilot uses one output-generation model, `gpt-4o-mini`, through the OpenAI API. All model settings are held constant across original and perturbed prompts so that observed differences can be attributed to repeated sampling or prompt perturbation rather than to changes in generation parameters. The pilot is intended to test the feasibility of the experimental pipeline and to generate preliminary results for RQ1. RQ2 is described as a planned extension because correctness evaluation has not yet been fully implemented.

## 1.1 Research Design

The study uses a repeated-generation and perturbation-based design. For each selected original prompt, the model generates multiple responses under identical settings. These repeated outputs are used to estimate the level of semantic variation that occurs without any prompt change. This estimate becomes the sampling-noise baseline.

After the baseline is estimated, selected prompts are modified using five perturbation types. The model then generates responses to the perturbed prompts using the same model and generation settings. The semantic similarity between original-prompt outputs and perturbed-prompt outputs is compared with the original sampling-noise baseline. This produces a noise-corrected estimate of perturbation effect.

The logic of the design is as follows:

```text
Observed output difference = ordinary sampling noise + possible perturbation effect
Noise-corrected perturbation effect = perturbation-related difference after accounting for sampling noise
```

This design is necessary because a single pairwise comparison between one original output and one perturbed output may overstate the effect of prompt perturbation. Some output difference would be expected even if the prompt had not been changed.

## 2. Shared Experimental Setup

The following sections describe the experimental materials and settings shared across RQ1 and the planned RQ2 analysis.

### 2.1 Datasets And Task Types

The study uses four task types: factual question answering, mathematical reasoning, code generation, and open-ended writing. These categories were selected because they represent different expected output structures and evaluation conditions.

| Task type | Dataset | Output characteristics | Rationale |
|---|---|---|---|
| Factual QA | SQuAD V2 | Context-question factual responses with reference answers | Tests robustness for constrained factual information using a literature-aligned QA dataset |
| Math reasoning | MATH / Hendrycks MATH | Step-by-step mathematical reasoning with objective final answers | Tests robustness for reasoning tasks using a literature-aligned mathematics benchmark |
| Code generation | HumanEvalPack Python | Executable or function-like code | Tests robustness where outputs may be textually different but functionally equivalent |
| Open-ended writing | Alpaca | Broad instruction-following responses | Tests robustness in tasks with flexible output space |

The distinction between mathematical reasoning and code generation is important. Although both tasks can be evaluated objectively, they should not be merged into one broad "verifiable task" category because they define output variation and correctness differently. In mathematical reasoning, the target is usually a final numeric answer, and variation often appears in the intermediate explanation or reasoning path. Two math outputs may use different steps but remain equivalent if they reach the same final answer. In code generation, the target is not a single textual answer but an executable or function-like artifact. Two code outputs may differ in variable names, helper functions, comments, type hints, control flow, or edge-case handling while still being functionally correct.

This distinction affects both the similarity analysis and the correctness analysis. For RQ1, the meaning of semantic similarity differs between math and code: lower textual or semantic similarity in code does not necessarily imply a worse or less correct output, because different implementations can solve the same task. For RQ2, the correctness criterion also differs: math reasoning can be evaluated by extracting and comparing the final answer, while code generation should be evaluated using functional correctness or test cases. Combining these two task types would mix different output structures and evaluation mechanisms, making the task-level baseline and perturbation effects harder to interpret.

Open-ended writing is included in RQ1 because semantic similarity and drift are meaningful for flexible generation tasks. However, it is not currently planned as a primary task type for RQ2 correctness analysis because there is no single objective answer key.

### 2.2 Prompt Sampling

Prompts were selected using stratified random sampling by task type. In this study, the task type is treated as the sampling stratum. Instead of combining all available prompts into one pool and drawing a simple random sample, the sampling procedure first separates the prompts into four strata: factual QA, math reasoning, code generation, and open-ended writing. A fixed number of prompts is then randomly selected from each stratum.

The algorithm can be summarized as:

```text
1. Define four task-type strata.
2. Load the corresponding benchmark dataset for each stratum.
3. Assign each available prompt a source index.
4. Set a fixed random seed for reproducibility.
5. Randomly sample the same number of prompts from each stratum.
6. Save the sampled prompts and their metadata.
```

This approach was chosen because the goal of the experiment is to compare output variation across task types. If simple random sampling were used across all datasets together, the final sample could contain more prompts from one task type than another, making task-level comparison less reliable. Stratified random sampling keeps the sample balanced across the four task categories.

The current formal sample contains:

```text
10 prompts per task type
4 task types
40 original prompts total
random seed = 20260623
```

The fixed random seed makes the sampling process reproducible. Each sampled item records the task type, dataset name, dataset split, source index, prompt text, reference answer when available, and random seed.

The sampling procedure is implemented in:

```text
src/06_sample_benchmark_prompts.py
prompts/rq1_sampled_original_prompts.csv
```

This sample size is appropriate for a pilot trial but should be expanded before making strong general claims about task-level differences.

### 2.3 Model And Generation Settings

The current pilot uses `gpt-4o-mini` as the output-generation model. This model was selected for practical reasons: it is available through the OpenAI API, affordable enough for repeated-generation experiments, and capable of handling all four selected task types. The current pilot does not compare multiple LLMs. Therefore, the model choice should be interpreted as a controlled pilot condition rather than as evidence that this model is optimal for the research question.

The generation settings are:

```text
temperature = 0.7
top_p = 0.9
max_output_tokens = not set
system_prompt = "You are a helpful assistant. Answer the user's prompt directly."
```

These settings are fixed across all original and perturbed prompts. The purpose of fixing these parameters is to control the generation environment. If different settings were used for different prompt conditions, the resulting output differences could be caused by model-parameter changes rather than by prompt variation.

Some of these values are pilot settings rather than empirically optimized values. In particular, `temperature = 0.7` and `top_p = 0.9` are pilot settings that allow non-deterministic generation while keeping the decoding setup constant. For subsequent data generation, no maximum output-token cap is manually set in the API request.

The generation configuration is stored in:

```text
config/rq1_generation_config.json
```

## 3. RQ1 Methodology: Sampling Noise And Prompt Perturbation

RQ1 asks whether the effects of different prompt perturbation types on output semantic similarity are consistent across task types or task-dependent after correcting for baseline sampling noise.

Operationally, RQ1 is divided into two methodological components:

```text
RQ1a: Estimate the sampling-noise baseline by repeatedly generating outputs from the same original prompt.
RQ1b: Estimate perturbation-induced semantic drift by comparing original-prompt outputs with perturbed-prompt outputs after applying the sampling-noise correction.
```

### 3.1 RQ1a: Measuring Sampling Noise

For RQ1a, each original prompt is submitted to the same LLM multiple times using the same generation settings. The outputs from the same prompt are compared with each other using semantic similarity. Higher within-prompt similarity indicates greater output stability, while lower within-prompt similarity indicates greater sampling noise.

Semantic similarity is measured using Sentence-BERT (SBERT), specifically the `sentence-transformers/all-MiniLM-L6-v2` model. Each generated output is encoded into a dense sentence embedding, and cosine similarity is computed between pairs of embeddings. This method is used because the study is concerned with meaning-level similarity rather than exact word overlap.

For a prompt with three generated outputs, the pairwise comparisons are:

```text
similarity(output_1, output_2)
similarity(output_1, output_3)
similarity(output_2, output_3)
```

The average of these pairwise similarities gives the within-prompt similarity score. The sampling-noise drift is then calculated as:

```text
sampling_noise_drift = 1 - mean_within_prompt_similarity
```

The task-level baseline is calculated by averaging the prompt-level drift values within each task type.

The current formal RQ1 baseline uses:

```text
40 original prompts
5 generations per prompt
200 total baseline outputs
```

The current Sentence-BERT RQ1a results are:

| Task type | Mean similarity | Mean sampling-noise drift | SD drift |
|---|---:|---:|---:|
| Factual QA | 0.949619 | 0.050381 | 0.049847 |
| Math reasoning | 0.939735 | 0.060265 | 0.031659 |
| Code generation | 0.942352 | 0.057648 | 0.026224 |
| Open-ended writing | 0.909664 | 0.090336 | 0.086416 |

These pilot results suggest that open-ended writing has the highest baseline output variability. This is consistent with the expectation that open-ended tasks allow a wider range of acceptable responses than factual QA or mathematical reasoning tasks.

### 3.2 Calibration Of Repeated Generations

Because the initial pilot used three generations per prompt, an additional calibration trial was conducted to examine whether the estimated sampling-noise baseline changes substantially when more repeated generations are used.

The calibration design was:

```text
2 prompts per task type
8 prompts total
10 generations per prompt
baseline recalculated at n = 3, 5, 7, and 10
```

The calibration results were:

| Task type | n=3 | n=5 | n=7 | n=10 |
|---|---:|---:|---:|---:|
| Code generation | 0.073214 | 0.065846 | 0.062641 | 0.058804 |
| Factual QA | 0.041099 | 0.035543 | 0.031909 | 0.031575 |
| Math reasoning | 0.077373 | 0.063475 | 0.061063 | 0.060591 |
| Open-ended writing | 0.178248 | 0.169233 | 0.165099 | 0.168513 |

The calibration suggests that three generations are acceptable for a low-cost pilot, but five generations per prompt would be preferable for the formal experiment. With three generations, each prompt produces only three pairwise similarity comparisons. With five generations, each prompt produces ten pairwise comparisons, which gives a more stable baseline estimate while keeping the API cost manageable.

### 3.3 RQ1b: Prompt Perturbation Types

RQ1b examines five perturbation types:

| Perturbation type | Definition |
|---|---|
| Paraphrasing | Meaning-preserving rewording |
| Reordering | Rearranging sentences or the order in which information is presented |
| Formatting changes | Converting text to a list or adding/removing polite phrases |
| Context injection | Adding background information unrelated to the question |
| Surface noise | Simulating unintentional spelling or punctuation errors that occur in real user input, such as missing letters or extra spaces |

These definitions follow the original proposal's perturbation design. The study focuses on natural everyday rephrasing rather than systematic adversarial attacks. Paraphrasing changes wording, reordering changes the order in which information is presented, formatting changes the prompt's presentation, context injection adds unrelated background information, and surface noise simulates realistic user-input errors.

In the current pilot, perturbations were manually constructed using the proposal definitions. All perturbed versions are manually checked to confirm that they are semantically equivalent to the original prompt. Versions that change the question itself, the answer target, or the task constraints should be regenerated, so that subsequent output differences reflect prompt wording or presentation rather than a changed task.

The current formal RQ1 perturbation stage uses:

```text
10 original prompts per task type
40 selected original prompts
5 perturbation types per prompt
200 perturbed prompts total
20 perturbed prompts
3 generations per perturbed prompt
60 perturbed outputs
```

### 3.4 Noise-Corrected Perturbation Effect

For each original prompt and perturbation condition, the semantic similarity between original-prompt outputs and perturbed-prompt outputs is compared with the original within-prompt baseline.

The same Sentence-BERT cosine similarity method is used for RQ1b so that the sampling-noise baseline and the perturbation effect are measured on the same scale.

The noise-corrected drift is calculated as:

```text
baseline_similarity = average similarity among repeated outputs from the original prompt
perturbation_similarity = average similarity between original-prompt outputs and perturbed-prompt outputs
noise_corrected_drift = baseline_similarity - perturbation_similarity
```

A positive value means that the perturbation produced outputs that were less similar to the original outputs than the original repeated outputs were to each other. A value close to zero means that the perturbation effect is similar to ordinary sampling noise. A negative value means that, in the pilot sample, the perturbed outputs were at least as similar to the original outputs as the original repeated outputs were to each other.

The current RQ1b pilot results are:

| Perturbation type | Code generation | Factual QA | Math reasoning | Open-ended writing |
|---|---:|---:|---:|---:|
| Paraphrasing | 0.041442 | -0.008114 | -0.005167 | -0.004696 |
| Reordering | 0.014932 | -0.020535 | 0.024498 | 0.004064 |
| Formatting changes | -0.009633 | 0.050575 | 0.066452 | 0.000425 |
| Context injection | 0.025948 | 0.062831 | 0.005024 | 0.066072 |
| Surface noise | -0.000620 | -0.010871 | -0.012020 | 0.006697 |

These preliminary results suggest that perturbation effects may be task-dependent. For example, formatting changes show a stronger effect for math reasoning in the pilot, while context injection shows stronger effects for factual QA and open-ended writing. However, because the RQ1b pilot currently uses only one original prompt per task type, these patterns should be treated as exploratory rather than conclusive.

## 4. RQ2 Planned Methodology: Semantic Drift And Correctness Change

RQ2 is planned as a follow-up analysis that examines whether semantic drift is associated with changes in answer correctness. This question is different from RQ1. RQ1 measures how much outputs change semantically, while RQ2 asks whether those changes matter for task success.

RQ2 should include only task types where correctness can be evaluated with a reasonably objective criterion:

| Task type | RQ2 inclusion | Planned correctness criterion |
|---|---|---|
| Factual QA | Included | Compare normalized model answer with reference answer or aliases |
| Math reasoning | Included | Extract and compare final numeric answer |
| Code generation | Included if executable tests are available | Run generated code against unit tests |
| Open-ended writing | Excluded from automatic correctness analysis | No single objective answer key |

For each original and perturbed output pair, the correctness label can be defined as:

```text
original_correct = whether the original-prompt output is correct
perturbed_correct = whether the perturbed-prompt output is correct
correctness_changed = original_correct != perturbed_correct
```

If sample size allows, correctness changes should also be separated into:

```text
correct -> incorrect
incorrect -> correct
no change
```

This distinction is useful because prompt perturbations may either harm or improve output correctness.

The planned RQ2 analysis can begin with a simple comparison:

```text
Compare average semantic drift for cases where correctness changed versus cases where correctness did not change.
```

If the sample size becomes large enough, a more formal model could be used:

```text
dependent variable = correctness_changed
main predictor = noise_corrected_drift
controls = task_type and perturbation_type
optional interaction = task_type * perturbation_type
```

At the current stage, RQ2 should be described as planned methodology rather than completed analysis.

## 5. Similarity Measure

The main similarity metric for RQ1 is Sentence-BERT cosine similarity using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Sentence-BERT converts each generated output into a dense semantic embedding. Cosine similarity is then used to estimate how semantically similar two outputs are. This approach is more appropriate than a simple bag-of-words method because two responses can be semantically similar even when they use different wording.

The procedure is:

```text
1. Encode each output text as a Sentence-BERT embedding.
2. Compute cosine similarity between pairs of embeddings.
3. Average pairwise similarities within each condition.
4. Convert similarity to drift where needed using 1 - similarity.
```

Earlier bag-of-words results should be treated only as preliminary pipeline checks, not as the main similarity analysis.

## 6. Experimental Procedure

The current pilot procedure is:

```text
1. Sample original prompts from four benchmark datasets using stratified random sampling.
2. Generate repeated outputs for each original prompt using the fixed model settings.
3. Compute the sampling-noise baseline for each prompt and task type.
4. Select one pilot prompt per task type for perturbation testing.
5. Construct five perturbed versions of each selected prompt.
6. Generate repeated outputs for each perturbed prompt.
7. Compute semantic similarity between original and perturbed outputs.
8. Calculate noise-corrected drift for each perturbation type and task type.
9. Compare perturbation patterns across task types.
```

The relevant scripts and files are:

```text
src/06_sample_benchmark_prompts.py
src/07_generate_rq1_outputs_openai.py
src/11_generate_rq1b_perturbed_outputs_openai.py
src/13_create_rq1_calibration_set.py
src/14_generate_rq1_calibration_outputs_openai.py
src/16_analyze_rq1_with_sentence_bert.py

prompts/rq1_sampled_original_prompts.csv
prompts/rq1b_pilot_perturbed_prompts.csv
prompts/rq1_calibration_prompts.csv

outputs/rq1_generations.csv
outputs/rq1b_pilot_perturbed_generations.csv
outputs/rq1_calibration_generations.csv
outputs/sbert_rq1a_noise_by_task.csv
outputs/sbert_rq1b_heatmap_noise_corrected_drift.csv
outputs/sbert_rq1_baseline_stability_by_task.csv
```

## 7. Limitations

The current methodology has several limitations. First, the pilot sample is small, especially for RQ1b, which currently uses only one original prompt per task type. Second, the study currently uses only one LLM, so the results cannot yet be generalized across models. Third, several generation parameters, including temperature and top-p, are fixed pilot settings rather than optimized values. Fourth, RQ2 correctness evaluation has not yet been implemented.

Despite these limitations, the pilot establishes a complete experimental pipeline for sampling prompts, generating repeated outputs, estimating sampling-noise baselines, constructing perturbations, computing Sentence-BERT similarity, and calculating noise-corrected drift. The next methodological step is to expand RQ1b to more prompts per task type and then implement RQ2 correctness labels for factual QA and math reasoning before extending to code-generation correctness tests.
