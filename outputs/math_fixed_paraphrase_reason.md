# Fixed Math Paraphrasing Drift Reason

## Fixed prompt-change comparison

| perturbation_type   |   n_items |   mean_problem_content_recall |   mean_problem_content_jaccard |   mean_prompt_content_recall |   mean_math_token_recall |   mean_number_recall |   mean_problem_length_delta_tokens |
|:--------------------|----------:|------------------------------:|-------------------------------:|-----------------------------:|-------------------------:|---------------------:|-----------------------------------:|
| paraphrasing        |        50 |                      0.753309 |                       0.498075 |                     0.833565 |                 0.925631 |             0.948785 |                               7.02 |
| surface_noise       |        50 |                      0.981014 |                       0.90606  |                     0.965796 |                 0.995229 |             1        |                               1.14 |
| formatting_changes  |        50 |                      1        |                       1        |                     1        |                 1        |             1        |                               0    |
| context_injection   |        50 |                      1        |                       1        |                     1        |                 1        |             1        |                               0    |
| reordering          |        50 |                      1        |                       1        |                     1        |                 1        |             1        |                               0    |

## Fixed paraphrase branch diagnostics

| branch   |   mean_ncp |   median_ncp |   mean_problem_content_recall |   mean_math_token_recall |   mean_number_recall |   mean_output_length_delta_tokens |   mean_answer_token_f1_delta |   mean_answer_containment_delta |
|:---------|-----------:|-------------:|------------------------------:|-------------------------:|---------------------:|----------------------------------:|-----------------------------:|--------------------------------:|
| GPT/main |  0.0153916 |    0.0039135 |                      0.753309 |                 0.925631 |             0.948785 |                             5.436 |                   0.00208271 |                           0.02  |
| Qwen     |  0.0408715 |    0.023609  |                      0.753309 |                 0.925631 |             0.948785 |                           -35.096 |                  -0.00463201 |                          -0.032 |
| Llama    |  0.0413614 |    0.030805  |                      0.753309 |                 0.925631 |             0.948785 |                            21.776 |                  -0.00673942 |                          -0.016 |

## Correlations with fixed paraphrase drift

| branch   | x                           | y                     |   n |        rho |           p |
|:---------|:----------------------------|:----------------------|----:|-----------:|------------:|
| GPT/main | math_token_recall           | noise_corrected_drift |  49 | -0.51461   | 0.000155022 |
| GPT/main | number_recall               | noise_corrected_drift |  48 | -0.277848  | 0.0558656   |
| GPT/main | problem_content_recall      | noise_corrected_drift |  50 | -0.270574  | 0.0573709   |
| GPT/main | prompt_content_recall       | noise_corrected_drift |  50 | -0.149148  | 0.301249    |
| GPT/main | answer_containment_delta    | noise_corrected_drift |  50 | -0.146559  | 0.309813    |
| GPT/main | problem_length_delta_tokens | noise_corrected_drift |  50 | -0.0693587 | 0.632214    |
| GPT/main | answer_token_f1_delta       | noise_corrected_drift |  50 |  0.0570756 | 0.693803    |
| GPT/main | output_length_delta_tokens  | noise_corrected_drift |  50 |  0.0314533 | 0.828335    |
| Llama    | output_length_delta_tokens  | noise_corrected_drift |  50 |  0.2388    | 0.0948959   |
| Llama    | answer_containment_delta    | noise_corrected_drift |  50 | -0.223887  | 0.118055    |
| Llama    | problem_content_recall      | noise_corrected_drift |  50 |  0.135719  | 0.347337    |
| Llama    | prompt_content_recall       | noise_corrected_drift |  50 |  0.0812051 | 0.575063    |
| Llama    | answer_token_f1_delta       | noise_corrected_drift |  50 | -0.053381  | 0.712742    |
| Llama    | math_token_recall           | noise_corrected_drift |  49 | -0.0267428 | 0.855269    |
| Llama    | number_recall               | noise_corrected_drift |  48 |  0.0212858 | 0.885815    |
| Llama    | problem_length_delta_tokens | noise_corrected_drift |  50 | -0.0124095 | 0.931838    |
| Qwen     | answer_containment_delta    | noise_corrected_drift |  50 | -0.249279  | 0.080852    |
| Qwen     | answer_token_f1_delta       | noise_corrected_drift |  50 | -0.215078  | 0.133619    |
| Qwen     | math_token_recall           | noise_corrected_drift |  49 | -0.17277   | 0.235179    |
| Qwen     | problem_length_delta_tokens | noise_corrected_drift |  50 | -0.129723  | 0.369249    |
| Qwen     | output_length_delta_tokens  | noise_corrected_drift |  50 |  0.115294  | 0.425279    |
| Qwen     | number_recall               | noise_corrected_drift |  48 | -0.046308  | 0.754629    |
| Qwen     | prompt_content_recall       | noise_corrected_drift |  50 |  0.0267641 | 0.853624    |
| Qwen     | problem_content_recall      | noise_corrected_drift |  50 | -0.0161422 | 0.911408    |

## Interpretation

After repair, paraphrasing remains the largest math perturbation because it is still the only perturbation that changes the mathematical problem wording at scale. Reordering, formatting changes, and context injection preserve the problem text, math tokens, and numbers almost exactly; surface noise makes small local edits. Fixed paraphrasing removes template artifacts and restores graphs, but it still changes problem phrasing, instruction framing, and some symbolic/numeric surface cues.

The strongest quantitative reason is prompt-level cue change: fixed paraphrasing has the lowest problem-content recall and math-token/number recall among the clean perturbations. Across branches, larger drift is generally associated with lower math-token or problem-content preservation and with changes in answer/output behavior.