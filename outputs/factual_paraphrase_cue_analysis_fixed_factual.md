# Fixed Factual QA Paraphrase Drift Analysis: outputs

- Rows: 50
- Mean noise-corrected drift: 0.084251
- Mean cue disruption: 0.322399
- Mean factual score delta: -0.045920
- Mean output length delta tokens: 15.576000

## Primary Spearman Correlations With Drift

| x                                      | y                     |   n |   spearman_rho |     p_value |
|:---------------------------------------|:----------------------|----:|---------------:|------------:|
| factual_score_delta                    | noise_corrected_drift |  50 |      -0.515824 | 0.000126127 |
| output_length_delta_tokens             | noise_corrected_drift |  50 |       0.459872 | 0.000779674 |
| capitalized_phrase_recall              | noise_corrected_drift |  29 |      -0.390366 | 0.0362957   |
| question_context_content_overlap_delta | noise_corrected_drift |  50 |       0.298468 | 0.0352601   |
| question_context_content_overlap_loss  | noise_corrected_drift |  50 |      -0.298468 | 0.0352601   |
| cue_disruption                         | noise_corrected_drift |  50 |       0.276098 | 0.0522781   |
| critical_cue_preservation              | noise_corrected_drift |  50 |      -0.276098 | 0.0522781   |
| question_content_recall                | noise_corrected_drift |  50 |      -0.224672 | 0.116736    |
| wh_word_preserved                      | noise_corrected_drift |  49 |      -0.176505 | 0.225058    |
| containment_rate_delta                 | noise_corrected_drift |  50 |      -0.156204 | 0.278693    |

## Selected Exploratory OLS Terms

| model                  | term                                  |   n |   r_squared |         coef |   std_error |    p_value | note                   |
|:-----------------------|:--------------------------------------|----:|------------:|-------------:|------------:|-----------:|:-----------------------|
| cue_correctness_length | cue_disruption                        |  50 |    0.34043  |  0.085506    | 0.0531822   | 0.10788    | OLS with HC3 robust SE |
| cue_correctness_length | factual_score_delta                   |  50 |    0.34043  | -0.31428     | 0.141733    | 0.0265948  | OLS with HC3 robust SE |
| cue_correctness_length | output_length_delta_tokens            |  50 |    0.34043  |  4.81608e-05 | 0.000243757 | 0.843376   | OLS with HC3 robust SE |
| expanded               | cue_disruption                        |  50 |    0.447626 | -0.00267908  | 0.0628941   | 0.966023   | OLS with HC3 robust SE |
| expanded               | question_content_recall               |  50 |    0.447626 | -0.152959    | 0.0654777   | 0.0194889  | OLS with HC3 robust SE |
| expanded               | question_context_content_overlap_loss |  50 |    0.447626 | -0.183246    | 0.135988    | 0.177813   | OLS with HC3 robust SE |
| expanded               | factual_score_delta                   |  50 |    0.447626 | -0.362257    | 0.111996    | 0.00121837 | OLS with HC3 robust SE |
| expanded               | output_length_delta_tokens            |  50 |    0.447626 | -0.000141787 | 0.000223376 | 0.525594   | OLS with HC3 robust SE |

## Interpretation

Spearman correlations are the primary evidence because n=50 is small. OLS models are exploratory and are included to check whether cue, correctness, and output-length signals survive in the same model.
