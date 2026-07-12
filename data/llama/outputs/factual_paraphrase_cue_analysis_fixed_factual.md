# Fixed Factual QA Paraphrase Drift Analysis: llama

- Rows: 50
- Mean noise-corrected drift: 0.084347
- Mean cue disruption: 0.322399
- Mean factual score delta: -0.031344
- Mean output length delta tokens: 13.528000

## Primary Spearman Correlations With Drift

| x                                      | y                     |   n |   spearman_rho |    p_value |
|:---------------------------------------|:----------------------|----:|---------------:|-----------:|
| output_length_delta_tokens             | noise_corrected_drift |  50 |     0.44422    | 0.00123066 |
| factual_score_delta                    | noise_corrected_drift |  50 |    -0.430921   | 0.00178385 |
| capitalized_phrase_recall              | noise_corrected_drift |  29 |    -0.26768    | 0.160359   |
| question_context_content_overlap_delta | noise_corrected_drift |  50 |     0.262649   | 0.0653682  |
| question_context_content_overlap_loss  | noise_corrected_drift |  50 |    -0.262649   | 0.0653682  |
| cue_disruption                         | noise_corrected_drift |  50 |     0.23278    | 0.103779   |
| critical_cue_preservation              | noise_corrected_drift |  50 |    -0.23278    | 0.103779   |
| question_content_recall                | noise_corrected_drift |  50 |    -0.221845   | 0.121533   |
| wh_word_preserved                      | noise_corrected_drift |  49 |    -0.151994   | 0.297151   |
| containment_rate_delta                 | noise_corrected_drift |  50 |    -0.00576432 | 0.968309   |

## Selected Exploratory OLS Terms

| model                  | term                                  |   n |   r_squared |         coef |   std_error |   p_value | note                   |
|:-----------------------|:--------------------------------------|----:|------------:|-------------:|------------:|----------:|:-----------------------|
| cue_correctness_length | cue_disruption                        |  50 |    0.132966 |  0.116604    | 0.0766898   |  0.128395 | OLS with HC3 robust SE |
| cue_correctness_length | factual_score_delta                   |  50 |    0.132966 |  0.0404923   | 0.324455    |  0.900681 | OLS with HC3 robust SE |
| cue_correctness_length | output_length_delta_tokens            |  50 |    0.132966 |  0.000467253 | 0.000681626 |  0.49303  | OLS with HC3 robust SE |
| expanded               | cue_disruption                        |  50 |    0.15006  |  0.10409     | 0.0883454   |  0.238708 | OLS with HC3 robust SE |
| expanded               | question_content_recall               |  50 |    0.15006  | -0.0135809   | 0.0833032   |  0.870495 | OLS with HC3 robust SE |
| expanded               | question_context_content_overlap_loss |  50 |    0.15006  | -0.151781    | 0.134032    |  0.257456 | OLS with HC3 robust SE |
| expanded               | factual_score_delta                   |  50 |    0.15006  |  0.0450557   | 0.332584    |  0.892239 | OLS with HC3 robust SE |
| expanded               | output_length_delta_tokens            |  50 |    0.15006  |  0.000436989 | 0.00079631  |  0.583165 | OLS with HC3 robust SE |

## Interpretation

Spearman correlations are the primary evidence because n=50 is small. OLS models are exploratory and are included to check whether cue, correctness, and output-length signals survive in the same model.
