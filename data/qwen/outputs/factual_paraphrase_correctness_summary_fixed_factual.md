# Factual Paraphrase Correctness Summary: qwen

- Rows: 50
- Mean original token-F1 score: 0.289535
- Mean paraphrase token-F1 score: 0.229621
- Mean factual score delta: -0.059914
- Mean original containment rate: 0.832000
- Mean paraphrase containment rate: 0.836000
- Mean containment rate delta: 0.004000
- Mean output length delta tokens: 26.980000

## Correlations With Noise-Corrected Drift

| metric                                 |   spearman_rho |     p_value |   n |
|:---------------------------------------|---------------:|------------:|----:|
| factual_score_delta                    |      -0.490838 | 0.000295723 |  50 |
| containment_rate_delta                 |      -0.207888 | 0.147425    |  50 |
| output_length_delta_tokens             |       0.415122 | 0.00272065  |  50 |
| cue_disruption                         |       0.250758 | 0.07901     |  50 |
| question_content_recall                |      -0.476557 | 0.000467724 |  50 |
| question_context_content_overlap_delta |       0.26658  | 0.0612964   |  50 |

## Highest Drift Items

| item_id         | reference_answer                                          |   noise_corrected_drift |   factual_score_delta |   containment_rate_delta |   output_length_delta_tokens |   cue_disruption |
|:----------------|:----------------------------------------------------------|------------------------:|----------------------:|-------------------------:|-----------------------------:|-----------------:|
| factual_qa_3339 | 2006                                                      |                0.580121 |             -0.69543  |                        0 |                         18.2 |         0.454545 |
| factual_qa_1601 | Switzerland and the Netherlands                           |                0.371976 |             -0.36174  |                        0 |                         42   |         1        |
| factual_qa_6578 | extension of the Florida East Coast Railway further south |                0.3694   |             -0.423966 |                        0 |                         55.6 |         0.5      |
| factual_qa_684  | skateboard                                                |                0.341089 |             -0.114282 |                        0 |                        104.4 |         0.555556 |
| factual_qa_7187 | taxation                                                  |                0.281941 |             -0.134042 |                        0 |                        176   |         0.333333 |
