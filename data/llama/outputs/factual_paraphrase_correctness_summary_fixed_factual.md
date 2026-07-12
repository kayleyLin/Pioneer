# Factual Paraphrase Correctness Summary: llama

- Rows: 50
- Mean original token-F1 score: 0.297960
- Mean paraphrase token-F1 score: 0.266616
- Mean factual score delta: -0.031344
- Mean original containment rate: 0.784000
- Mean paraphrase containment rate: 0.800000
- Mean containment rate delta: 0.016000
- Mean output length delta tokens: 13.528000

## Correlations With Noise-Corrected Drift

| metric                                 |   spearman_rho |    p_value |   n |
|:---------------------------------------|---------------:|-----------:|----:|
| factual_score_delta                    |    -0.430921   | 0.00178385 |  50 |
| containment_rate_delta                 |    -0.00576432 | 0.968309   |  50 |
| output_length_delta_tokens             |     0.44422    | 0.00123066 |  50 |
| cue_disruption                         |     0.23278    | 0.103779   |  50 |
| question_content_recall                |    -0.221845   | 0.121533   |  50 |
| question_context_content_overlap_delta |     0.262649   | 0.0653682  |  50 |

## Highest Drift Items

| item_id         | reference_answer                                                                 |   noise_corrected_drift |   factual_score_delta |   containment_rate_delta |   output_length_delta_tokens |   cue_disruption |
|:----------------|:---------------------------------------------------------------------------------|------------------------:|----------------------:|-------------------------:|-----------------------------:|-----------------:|
| factual_qa_3339 | 2006                                                                             |                0.368429 |             0.516223  |                        0 |                        -16.4 |        0.454545  |
| factual_qa_8650 | 2001                                                                             |                0.298569 |            -0.104575  |                        0 |                          8   |        0.0666667 |
| factual_qa_1601 | Switzerland and the Netherlands                                                  |                0.293422 |            -0.296388  |                        0 |                         33.4 |        1         |
| factual_qa_684  | skateboard                                                                       |                0.245901 |            -0.11448   |                        0 |                         90.4 |        0.555556  |
| factual_qa_5525 | the head of government would be acting in her or his capacity as public official |                0.198642 |            -0.0131884 |                        0 |                         14.2 |        0.916667  |
