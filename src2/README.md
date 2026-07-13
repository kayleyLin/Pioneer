# Source Script Notes

## `build_new_llm_model_figures.py`

Purpose: refresh `token_analysis_figures/new_llm_model/fig1` to `fig6` using the
new Llama 3.3 70B n50 factual QA paraphrasing data.

The script implements the plan in:

```text
token_analysis_figures/figures_plan.md
```

It reads the Llama 3.3 70B four-task n50 generation files:

```text
llama/llama33_70b_instruct_turbo_150case/outputs/rq1_llama33_70b_original_generations_n50_four_task.csv
llama/llama33_70b_instruct_turbo_150case/outputs/rq1_llama33_70b_perturbed_generations_n50_four_task.csv
```

It extracts the factual QA original and factual QA paraphrasing subsets, recomputes
the Llama 3.3 70B fixed factual paraphrase text-feature pipeline, and then redraws
the cross-model token-analysis figures with:

```text
GPT/main
Llama 3.3 70B
Qwen
```

Outputs are written to:

```text
token_analysis_figures/new_llm_model/
token_analysis_figures/new_llm_model/intermediate/
```

Expected final figures:

```text
fig1_cross_model_feature_correlation_heatmap.png
fig2_output_vs_prompt_feature_strength.png
fig3_output_edit_distance_vs_drift.png
fig4_output_length_delta_vs_drift.png
fig5_reference_f1_delta_vs_drift.png
fig6_containment_delta_vs_drift.png
```

Run from the project root:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python src\build_new_llm_model_figures.py
```

The offline environment variables are recommended after the first successful
Sentence-BERT model load, so the script uses the local Hugging Face cache instead
of attempting network access. If the model is not cached, run without those
variables in an environment with Hugging Face access.

The script does not overwrite the old `token_analysis_figures/fig1` to `fig6`
files in the parent directory.
