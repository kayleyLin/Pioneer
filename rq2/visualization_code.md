# RQ2 Visualization Code

This document records the visualization script used to generate candidate RQ1/RQ2 heatmaps.

## Script

```text
src/27_create_rq_heatmaps.py
```

Run command:

```bash
/opt/anaconda3/bin/python src/27_create_rq_heatmaps.py
```

To write a versioned figure set without overwriting older figures:

```bash
/opt/anaconda3/bin/python src/27_create_rq_heatmaps.py \
  --figures-dir /Users/wenfenglin/Desktop/Pioneer/figures/2.0
```

## Output Folder

All generated figures are saved to:

```text
figures/
```

Current rerun output folder:

```text
figures/2.0/
```

## Input Files

The script reads:

```text
rq2/outputs/rq2_formal_available_drift_performance_summary.csv
rq2/outputs/rq2_formal_available_performance_change_summary.csv
rq2/outputs/rq2_stats_correlations.csv
```

## Generated Figures

| Figure file | RQ label | What it shows |
|---|---|---|
| `rq1_rq2_noise_corrected_drift_heatmap.png` | RQ1/RQ2 | Mean noise-corrected semantic drift by task and perturbation |
| `rq2_performance_change_heatmap.png` | RQ2 | Mean absolute performance change by task and perturbation |
| `rq2_pdr_heatmap.png` | RQ2 | Mean performance drop rate by task and perturbation |
| `rq2_pearson_drift_performance_correlation_heatmap.png` | RQ2 | Pearson correlation between semantic drift and performance change |
| `rq2_spearman_drift_performance_correlation_heatmap.png` | RQ2 | Spearman correlation between semantic drift and performance change |
| `rq2_performance_vs_drift_contrast_heatmap.png` | RQ2 exploratory | Standardized contrast between performance change and semantic drift |

## Recommended Figures For Paper

Most useful RQ2 candidates:

```text
rq2_performance_change_heatmap.png
rq2_pearson_drift_performance_correlation_heatmap.png
rq2_spearman_drift_performance_correlation_heatmap.png
```

The drift heatmap overlaps with RQ1, so it is best used only as a predictor-reference figure or paired with the RQ2 performance heatmap.

The contrast heatmap is exploratory. Use it only if the paper needs to highlight cases where performance changes more or less than semantic drift would suggest.
