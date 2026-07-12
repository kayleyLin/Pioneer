# Factual Text-Level Feature Driver Analysis (Qwen)

Run date: 2026-07-10

## Purpose

This analysis tests whether observable text-level changes explain the residual fixed factual QA paraphrase drift. It is designed to strengthen or qualify the current scope/style explanation before moving to probability or attention probes.

## Step 1. Feature Base Validation

- Rows: 50
- Unique items: 50
- Missing `noise_corrected_drift`: 0
- Missing original/paraphrased questions: 0
- Mean fixed factual QA NCP: 0.131119

Conclusion: the `qwen` fixed factual QA table is complete enough for text-level driver analysis.

## Step 2. Prompt-Side Rewrite Features

- Mean normalized question token edit distance: 0.6755
- Mean question length delta: 8.220 tokens
- Mean normalized prompt token edit distance: 0.0908
- `question_token_edit_distance_norm` vs drift: rho=0.3691, p=0.00834

Conclusion: prompt/question rewrite magnitude is measured directly. Its explanatory value should be judged against the output-side features below.

## Step 3. Output-Side Edit And Expansion Features

- Mean output length delta: 26.980 tokens
- Mean output length ratio: 3.6177
- Mean all-pairs normalized output token edit distance: 0.6407
- `output_length_delta_tokens` vs drift: rho=0.4151, p=0.002721

Conclusion: this step quantifies whether drift is visible as generated-output expansion and direct output text divergence.

## Step 4. Scope / Style Proxy Features

- `factual_score_delta` vs drift: rho=-0.4908, p=0.0002957
- `containment_rate_delta` vs drift: rho=-0.2079, p=0.1474
- `answer_scope_proxy` vs drift: rho=0.3776, p=0.006861

Conclusion: the proxy separates compact-reference-answer loss and output expansion from simple reference containment failure.

## Step 5. Correlation Ranking

Top positive relationships with drift:

| Feature | rho | p | n |
|---|---:|---:|---:|
| `median_output_token_edit_distance_norm` | 0.6913 | 2.734e-08 | 50 |
| `mean_output_token_edit_distance_norm` | 0.6779 | 6.366e-08 | 50 |
| `mean_output_char_edit_distance_norm` | 0.5542 | 2.974e-05 | 50 |
| `output_length_ratio` | 0.5211 | 0.0001044 | 50 |
| `answer_compactness_loss_proxy` | 0.4908 | 0.0002957 | 50 |
| `output_length_delta_tokens` | 0.4151 | 0.002721 | 50 |

Top negative relationships with drift:

| Feature | rho | p | n |
|---|---:|---:|---:|
| `factual_score_delta` | -0.4908 | 0.0002957 | 50 |
| `question_content_recall` | -0.4766 | 0.0004677 | 50 |
| `containment_rate_delta` | -0.2079 | 0.1474 | 50 |
| `prompt_char_edit_distance_norm` | 0.2413 | 0.09139 | 50 |
| `cue_disruption` | 0.2508 | 0.07901 | 50 |
| `question_context_content_overlap_delta` | 0.2666 | 0.0613 | 50 |

Mean absolute rho for output-side features: 0.4842
Mean absolute rho for prompt-side features: 0.3224

Conclusion: compare these two means and the ranked features to decide whether generated-answer form explains more than prompt rewrite magnitude.

## Step 6. Exploratory Regressions

Minimal text-driver model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `output_length_delta_tokens` | 0.0001 | 0.5174 | 0.3456 |
| `factual_score_delta` | -0.2774 | 0.1785 | 0.3456 |
| `question_token_edit_distance_norm` | 0.0866 | 0.2982 | 0.3456 |
| `cue_disruption` | 0.0657 | 0.2555 | 0.3456 |

Scope/style model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `answer_scope_proxy` | 0.0224 | 0.1015 | 0.3159 |
| `question_content_recall` | -0.2384 | 0.0004292 | 0.3159 |
| `question_context_content_overlap_delta` | 0.1904 | 0.2154 | 0.3159 |

Output-distance model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `mean_output_token_edit_distance_norm` | 0.2584 | 1.146e-06 | 0.3267 |
| `question_token_edit_distance_norm` | 0.0392 | 0.639 | 0.3267 |
| `containment_rate_delta` | 0.0167 | 0.8619 | 0.3267 |

Conclusion: these regressions are exploratory because n=50. Use coefficient direction, R2, and consistency with Spearman patterns rather than treating p-values as decisive.

## Files Written

- `qwen\outputs\factual_text_feature_base_fixed_factual.csv`
- `qwen\outputs\factual_text_feature_driver_correlations_fixed_factual.csv`
- `qwen\outputs\factual_text_feature_driver_regressions_fixed_factual.csv`
- `qwen\outputs\factual_text_feature_driver_summary_fixed_factual.md`
