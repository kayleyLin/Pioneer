# Factual Paraphrase Correctness Summary: outputs

- Rows: 50
- Mean original token-F1 score: 0.239293
- Mean paraphrase token-F1 score: 0.193373
- Mean factual score delta: -0.045920
- Mean original containment rate: 0.812000
- Mean paraphrase containment rate: 0.788000
- Mean containment rate delta: -0.024000
- Mean output length delta tokens: 15.576000

## Correlations With Noise-Corrected Drift

| metric                                 |   spearman_rho |     p_value |   n |
|:---------------------------------------|---------------:|------------:|----:|
| factual_score_delta                    |      -0.515824 | 0.000126127 |  50 |
| containment_rate_delta                 |      -0.156204 | 0.278693    |  50 |
| output_length_delta_tokens             |       0.459872 | 0.000779674 |  50 |
| cue_disruption                         |       0.276098 | 0.0522781   |  50 |
| question_content_recall                |      -0.224672 | 0.116736    |  50 |
| question_context_content_overlap_delta |       0.298468 | 0.0352601   |  50 |

## Highest Drift Items

| item_id         | reference_answer                                                                 |   noise_corrected_drift |   factual_score_delta |   containment_rate_delta |   output_length_delta_tokens |   cue_disruption |
|:----------------|:---------------------------------------------------------------------------------|------------------------:|----------------------:|-------------------------:|-----------------------------:|-----------------:|
| factual_qa_1601 | Switzerland and the Netherlands                                                  |                0.276404 |            -0.291604  |                        0 |                         30.4 |         1        |
| factual_qa_2022 | Matthew Murray                                                                   |                0.264717 |            -0.278363  |                        0 |                         10.8 |         0.111111 |
| factual_qa_5525 | the head of government would be acting in her or his capacity as public official |                0.254322 |            -0.0258167 |                        0 |                         16   |         0.916667 |
| factual_qa_7187 | taxation                                                                         |                0.249991 |            -0.11088   |                        0 |                         77.6 |         0.333333 |
| factual_qa_4277 | circuit switching                                                                |                0.225829 |            -0.214286  |                        0 |                          6   |         0.5      |
