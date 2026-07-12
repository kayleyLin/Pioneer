# Fixed Factual QA Paraphrase Drift Analysis: qwen

- Rows: 50
- Mean noise-corrected drift: 0.131119
- Mean cue disruption: 0.322399
- Mean factual score delta: -0.059914
- Mean output length delta tokens: 26.980000

## Primary Spearman Correlations With Drift

| x                                      | y                     |   n |   spearman_rho |     p_value |
|:---------------------------------------|:----------------------|----:|---------------:|------------:|
| factual_score_delta                    | noise_corrected_drift |  50 |     -0.490838  | 0.000295723 |
| question_content_recall                | noise_corrected_drift |  50 |     -0.476557  | 0.000467724 |
| output_length_delta_tokens             | noise_corrected_drift |  50 |      0.415122  | 0.00272065  |
| question_context_content_overlap_delta | noise_corrected_drift |  50 |      0.26658   | 0.0612964   |
| question_context_content_overlap_loss  | noise_corrected_drift |  50 |     -0.26658   | 0.0612964   |
| cue_disruption                         | noise_corrected_drift |  50 |      0.250758  | 0.07901     |
| critical_cue_preservation              | noise_corrected_drift |  50 |     -0.250758  | 0.07901     |
| containment_rate_delta                 | noise_corrected_drift |  50 |     -0.207888  | 0.147425    |
| capitalized_phrase_recall              | noise_corrected_drift |  29 |     -0.184588  | 0.33777     |
| wh_word_preserved                      | noise_corrected_drift |  49 |     -0.0621034 | 0.671631    |

## Selected Exploratory OLS Terms

| model                  | term                                  |   n |   r_squared |         coef |   std_error |    p_value | note                   |
|:-----------------------|:--------------------------------------|----:|------------:|-------------:|------------:|-----------:|:-----------------------|
| cue_correctness_length | cue_disruption                        |  50 |    0.326973 |  0.0855512   | 0.058268    | 0.14204    | OLS with HC3 robust SE |
| cue_correctness_length | factual_score_delta                   |  50 |    0.326973 | -0.288587    | 0.183833    | 0.116453   | OLS with HC3 robust SE |
| cue_correctness_length | output_length_delta_tokens            |  50 |    0.326973 |  0.000145779 | 0.00017454  | 0.403592   | OLS with HC3 robust SE |
| expanded               | cue_disruption                        |  50 |    0.445296 | -0.0614705   | 0.0620616   | 0.321942   | OLS with HC3 robust SE |
| expanded               | question_content_recall               |  50 |    0.445296 | -0.262317    | 0.0920842   | 0.00439029 | OLS with HC3 robust SE |
| expanded               | question_context_content_overlap_loss |  50 |    0.445296 | -0.270443    | 0.168903    | 0.109338   | OLS with HC3 robust SE |
| expanded               | factual_score_delta                   |  50 |    0.445296 | -0.316797    | 0.184834    | 0.0865369  | OLS with HC3 robust SE |
| expanded               | output_length_delta_tokens            |  50 |    0.445296 | -2.35726e-05 | 0.000280617 | 0.933054   | OLS with HC3 robust SE |

## Interpretation

Spearman correlations are the primary evidence because n=50 is small. OLS models are exploratory and are included to check whether cue, correctness, and output-length signals survive in the same model.
