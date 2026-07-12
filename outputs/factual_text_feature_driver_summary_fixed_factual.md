# Factual Text-Level Feature Driver Analysis (GPT/main)

Run date: 2026-07-10

## Purpose

This analysis tests whether observable text-level changes explain the residual fixed factual QA paraphrase drift. It is designed to strengthen or qualify the current scope/style explanation before moving to probability or attention probes.

## Step 1. Feature Base Validation

- Rows: 50
- Unique items: 50
- Missing `noise_corrected_drift`: 0
- Missing original/paraphrased questions: 0
- Mean fixed factual QA NCP: 0.084251

Conclusion: the `outputs/` fixed factual QA table is complete enough for text-level driver analysis.

## Step 2. Prompt-Side Rewrite Features

- Mean normalized question token edit distance: 0.6755
- Mean question length delta: 8.220 tokens
- Mean normalized prompt token edit distance: 0.0908
- `question_token_edit_distance_norm` vs drift: rho=0.3254, p=0.02114

Conclusion: prompt/question rewrite magnitude is measured directly. Its explanatory value should be judged against the output-side features below.

## Step 3. Output-Side Edit And Expansion Features

- Mean output length delta: 15.576 tokens
- Mean output length ratio: 1.5307
- Mean all-pairs normalized output token edit distance: 0.6090
- `output_length_delta_tokens` vs drift: rho=0.4599, p=0.0007797

Conclusion: this step quantifies whether drift is visible as generated-output expansion and direct output text divergence.

## Step 4. Scope / Style Proxy Features

- `factual_score_delta` vs drift: rho=-0.5158, p=0.0001261
- `containment_rate_delta` vs drift: rho=-0.1562, p=0.2787
- `answer_scope_proxy` vs drift: rho=0.4281, p=0.001925

Conclusion: the proxy separates compact-reference-answer loss and output expansion from simple reference containment failure.

## Step 5. Correlation Ranking

Top positive relationships with drift:

| Feature | rho | p | n |
|---|---:|---:|---:|
| `median_output_token_edit_distance_norm` | 0.5731 | 1.366e-05 | 50 |
| `mean_output_token_edit_distance_norm` | 0.5531 | 3.102e-05 | 50 |
| `output_length_ratio` | 0.5363 | 5.966e-05 | 50 |
| `answer_compactness_loss_proxy` | 0.5158 | 0.0001261 | 50 |
| `style_expansion_proxy` | 0.4599 | 0.0007797 | 50 |
| `output_length_delta_tokens` | 0.4599 | 0.0007797 | 50 |

Top negative relationships with drift:

| Feature | rho | p | n |
|---|---:|---:|---:|
| `factual_score_delta` | -0.5158 | 0.0001261 | 50 |
| `question_content_recall` | -0.2247 | 0.1167 | 50 |
| `containment_rate_delta` | -0.1562 | 0.2787 | 50 |
| `prompt_char_edit_distance_norm` | 0.2231 | 0.1193 | 50 |
| `cue_disruption` | 0.2761 | 0.05228 | 50 |
| `question_context_content_overlap_delta` | 0.2985 | 0.03526 | 50 |

Mean absolute rho for output-side features: 0.4578
Mean absolute rho for prompt-side features: 0.3064

Conclusion: compare these two means and the ranked features to decide whether generated-answer form explains more than prompt rewrite magnitude.

## Step 6. Exploratory Regressions

Minimal text-driver model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `output_length_delta_tokens` | -0.0000 | 0.8662 | 0.3882 |
| `factual_score_delta` | -0.2861 | 0.02876 | 0.3882 |
| `question_token_edit_distance_norm` | 0.0879 | 0.04402 | 0.3882 |
| `cue_disruption` | 0.0662 | 0.2118 | 0.3882 |

Scope/style model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `answer_scope_proxy` | 0.0116 | 0.08929 | 0.2652 |
| `question_content_recall` | -0.1275 | 0.02798 | 0.2652 |
| `question_context_content_overlap_delta` | 0.1728 | 0.1843 | 0.2652 |

Output-distance model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `mean_output_token_edit_distance_norm` | 0.1557 | 6.375e-05 | 0.3217 |
| `question_token_edit_distance_norm` | 0.0544 | 0.2378 | 0.3217 |
| `containment_rate_delta` | -0.0376 | 0.4306 | 0.3217 |

Conclusion: these regressions are exploratory because n=50. Use coefficient direction, R2, and consistency with Spearman patterns rather than treating p-values as decisive.

## Files Written

- `outputs\factual_text_feature_base_fixed_factual.csv`
- `outputs\factual_text_feature_driver_correlations_fixed_factual.csv`
- `outputs\factual_text_feature_driver_regressions_fixed_factual.csv`
- `outputs\factual_text_feature_driver_summary_fixed_factual.md`
