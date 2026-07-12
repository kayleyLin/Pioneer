# Factual Text-Level Feature Driver Analysis (Llama)

Run date: 2026-07-10

## Purpose

This analysis tests whether observable text-level changes explain the residual fixed factual QA paraphrase drift. It is designed to strengthen or qualify the current scope/style explanation before moving to probability or attention probes.

## Step 1. Feature Base Validation

- Rows: 50
- Unique items: 50
- Missing `noise_corrected_drift`: 0
- Missing original/paraphrased questions: 0
- Mean fixed factual QA NCP: 0.084347

Conclusion: the `llama` fixed factual QA table is complete enough for text-level driver analysis.

## Step 2. Prompt-Side Rewrite Features

- Mean normalized question token edit distance: 0.6755
- Mean question length delta: 8.220 tokens
- Mean normalized prompt token edit distance: 0.0908
- `question_token_edit_distance_norm` vs drift: rho=0.3860, p=0.005636

Conclusion: prompt/question rewrite magnitude is measured directly. Its explanatory value should be judged against the output-side features below.

## Step 3. Output-Side Edit And Expansion Features

- Mean output length delta: 13.528 tokens
- Mean output length ratio: 1.6502
- Mean all-pairs normalized output token edit distance: 0.5672
- `output_length_delta_tokens` vs drift: rho=0.4442, p=0.001231

Conclusion: this step quantifies whether drift is visible as generated-output expansion and direct output text divergence.

## Step 4. Scope / Style Proxy Features

- `factual_score_delta` vs drift: rho=-0.4309, p=0.001784
- `containment_rate_delta` vs drift: rho=-0.0058, p=0.9683
- `answer_scope_proxy` vs drift: rho=0.4160, p=0.002662

Conclusion: the proxy separates compact-reference-answer loss and output expansion from simple reference containment failure.

## Step 5. Correlation Ranking

Top positive relationships with drift:

| Feature | rho | p | n |
|---|---:|---:|---:|
| `mean_output_char_edit_distance_norm` | 0.6666 | 1.258e-07 | 50 |
| `mean_output_token_edit_distance_norm` | 0.6634 | 1.523e-07 | 50 |
| `median_output_token_edit_distance_norm` | 0.6275 | 1.079e-06 | 50 |
| `prompt_token_edit_distance_norm` | 0.4755 | 0.0004834 | 50 |
| `output_length_delta_tokens` | 0.4442 | 0.001231 | 50 |
| `output_length_ratio` | 0.4440 | 0.00124 | 50 |

Top negative relationships with drift:

| Feature | rho | p | n |
|---|---:|---:|---:|
| `factual_score_delta` | -0.4309 | 0.001784 | 50 |
| `question_content_recall` | -0.2218 | 0.1215 | 50 |
| `containment_rate_delta` | -0.0058 | 0.9683 | 50 |
| `cue_disruption` | 0.2328 | 0.1038 | 50 |
| `question_context_content_overlap_delta` | 0.2626 | 0.06537 | 50 |
| `question_char_edit_distance_norm` | 0.2870 | 0.04332 | 50 |

Mean absolute rho for output-side features: 0.4573
Mean absolute rho for prompt-side features: 0.3465

Conclusion: compare these two means and the ranked features to decide whether generated-answer form explains more than prompt rewrite magnitude.

## Step 6. Exploratory Regressions

Minimal text-driver model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `output_length_delta_tokens` | 0.0004 | 0.5717 | 0.1524 |
| `factual_score_delta` | 0.0615 | 0.8509 | 0.1524 |
| `question_token_edit_distance_norm` | 0.0678 | 0.2813 | 0.1524 |
| `cue_disruption` | 0.1005 | 0.1765 | 0.1524 |

Scope/style model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `answer_scope_proxy` | 0.0046 | 0.8468 | 0.0919 |
| `question_content_recall` | -0.0942 | 0.2698 | 0.0919 |
| `question_context_content_overlap_delta` | 0.2082 | 0.1973 | 0.0919 |

Output-distance model:

| Term | coef | p | R2 |
|---|---:|---:|---:|
| `mean_output_token_edit_distance_norm` | 0.2126 | 8.612e-05 | 0.3473 |
| `question_token_edit_distance_norm` | 0.0111 | 0.866 | 0.3473 |
| `containment_rate_delta` | -0.0091 | 0.8315 | 0.3473 |

Conclusion: these regressions are exploratory because n=50. Use coefficient direction, R2, and consistency with Spearman patterns rather than treating p-values as decisive.

## Files Written

- `outputs\llama_factual_text_feature_base_fixed_factual.csv`
- `outputs\llama_factual_text_feature_driver_correlations_fixed_factual.csv`
- `outputs\llama_factual_text_feature_driver_regressions_fixed_factual.csv`
- `outputs\llama_factual_text_feature_driver_summary_fixed_factual.md`
