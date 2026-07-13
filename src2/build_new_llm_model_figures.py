"""Build token-analysis fig1-fig6 with Llama 3.3 70B n50 data.

This script follows token_analysis_figures/figures_plan.md. It leaves the old
token_analysis_figures outputs untouched and writes new artifacts to
token_analysis_figures/new_llm_model/.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "token_analysis_figures" / "new_llm_model"
INTERMEDIATE = OUT / "intermediate"
LLAMA33_OUT = ROOT / "llama" / "llama33_70b_instruct_turbo_150case" / "outputs"

LLAMA33_ORIGINAL_FOUR_TASK = LLAMA33_OUT / "rq1_llama33_70b_original_generations_n50_four_task.csv"
LLAMA33_PERTURBED_FOUR_TASK = LLAMA33_OUT / "rq1_llama33_70b_perturbed_generations_n50_four_task.csv"

WORKING_ORIGINAL = INTERMEDIATE / "rq1_llama33_70b_original_generations_n50_factual_qa.csv"
WORKING_PARAPHRASE = INTERMEDIATE / "rq1_llama33_70b_perturbed_generations_n50_factual_qa_paraphrasing.csv"
CELL_EFFECTS = INTERMEDIATE / "sbert_rq1_n50_fixed_factual_paraphrase_effects_by_item_llama33_70b.csv"
ITEM_TABLE = INTERMEDIATE / "factual_paraphrase_item_table_fixed_factual_llama33_70b.csv"
CUE_METRICS = INTERMEDIATE / "factual_paraphrase_cue_metrics_fixed_factual_llama33_70b.csv"
CORRECTNESS = INTERMEDIATE / "factual_paraphrase_correctness_by_item_fixed_factual_llama33_70b.csv"
CORRECTNESS_SUMMARY = INTERMEDIATE / "factual_paraphrase_correctness_summary_fixed_factual_llama33_70b.md"

TEXT_BASE = INTERMEDIATE / "factual_text_feature_base_fixed_factual_llama33_70b.csv"
TEXT_CORR = INTERMEDIATE / "factual_text_feature_driver_correlations_fixed_factual_llama33_70b.csv"
TEXT_REG = INTERMEDIATE / "factual_text_feature_driver_regressions_fixed_factual_llama33_70b.csv"
TEXT_SUMMARY = INTERMEDIATE / "factual_text_feature_driver_summary_fixed_factual_llama33_70b.md"

GPT_BASE = ROOT / "outputs" / "factual_text_feature_base_fixed_factual.csv"
GPT_CORR = ROOT / "outputs" / "factual_text_feature_driver_correlations_fixed_factual.csv"
QWEN_BASE = ROOT / "qwen" / "outputs" / "factual_text_feature_base_fixed_factual.csv"
QWEN_CORR = ROOT / "qwen" / "outputs" / "factual_text_feature_driver_correlations_fixed_factual.csv"

PERTURBED_PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_factual_qa_fixed.csv"

MODELS = ["GPT/main", "Llama 3.3 70B", "Qwen"]
MODEL_COLORS = {
    "GPT/main": "#4C78A8",
    "Llama 3.3 70B": "#F58518",
    "Qwen": "#54A24B",
}

KEY_FEATURES = [
    "mean_output_token_edit_distance_norm",
    "median_output_token_edit_distance_norm",
    "output_length_delta_tokens",
    "output_length_ratio",
    "factual_score_delta",
    "containment_rate_delta",
    "question_token_edit_distance_norm",
    "cue_disruption",
]

FEATURE_LABELS = {
    "mean_output_token_edit_distance_norm": "Mean output token edit distance",
    "median_output_token_edit_distance_norm": "Median output token edit distance",
    "output_length_delta_tokens": "Output length delta",
    "output_length_ratio": "Output length ratio",
    "factual_score_delta": "Reference token-F1 delta",
    "containment_rate_delta": "Reference containment delta",
    "question_token_edit_distance_norm": "Question token edit distance",
    "cue_disruption": "Cue disruption",
    "noise_corrected_drift": "Noise-corrected drift",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sample_sets(df: pd.DataFrame, key_cols: list[str], label: str) -> None:
    bad = []
    for key, group in df.groupby(key_cols):
        samples = set(group["sample_id"].astype(str))
        if samples != {"1", "2", "3", "4", "5"}:
            bad.append((key, samples))
    if bad:
        raise SystemExit(f"{label}: {len(bad)} groups do not have sample_id 1..5")


def extract_working_generation_files() -> None:
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(LLAMA33_ORIGINAL_FOUR_TASK)
    perturbed = pd.read_csv(LLAMA33_PERTURBED_FOUR_TASK)

    original_factual = original[original["task_type"] == "factual_qa"].copy()
    paraphrase = perturbed[
        (perturbed["task_type"] == "factual_qa")
        & (perturbed["perturbation_type"] == "paraphrasing")
    ].copy()

    if len(original_factual) != 250 or original_factual["item_id"].nunique() != 50:
        raise SystemExit("Llama 3.3 70B factual original subset is not 50 x 5.")
    if len(paraphrase) != 250 or paraphrase["item_id"].nunique() != 50:
        raise SystemExit("Llama 3.3 70B factual paraphrase subset is not 50 x 5.")
    if original_factual["output_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("Llama 3.3 70B factual original subset has empty output_text.")
    if paraphrase["output_text"].fillna("").astype(str).str.strip().eq("").any():
        raise SystemExit("Llama 3.3 70B factual paraphrase subset has empty output_text.")

    validate_sample_sets(original_factual, ["item_id"], "original factual")
    validate_sample_sets(paraphrase, ["item_id", "perturbation_type"], "paraphrase factual")

    original_factual.to_csv(WORKING_ORIGINAL, index=False)
    paraphrase.to_csv(WORKING_PARAPHRASE, index=False)


def recompute_sbert_cell() -> None:
    mod = load_module("recompute_llama_cell", ROOT / "src" / "42_recompute_llama_fixed_factual_paraphrase_cell.py")
    mod.ORIGINAL = WORKING_ORIGINAL
    mod.FIXED_PARAPHRASE = WORKING_PARAPHRASE
    mod.CELL_EFFECTS = CELL_EFFECTS
    cell = mod.recompute_cell()
    if len(cell) != 50 or cell["item_id"].nunique() != 50:
        raise SystemExit("SBERT cell recompute did not produce 50 factual QA rows.")


def build_item_table() -> None:
    mod = load_module("build_item_table", ROOT / "src" / "44_build_factual_paraphrase_analysis_table.py")
    branch = "llama33_70b"
    mod.BRANCHES[branch] = {
        "output_dir": INTERMEDIATE,
        "output_name": ITEM_TABLE.name,
        "original_generations": WORKING_ORIGINAL,
        "perturbed_generations": WORKING_PARAPHRASE,
        "effects": CELL_EFFECTS,
    }
    table = mod.write_branch(branch, PERTURBED_PROMPTS)
    if len(table) != 50:
        raise SystemExit("Item table did not produce 50 rows.")


def compute_cue_metrics() -> None:
    mod = load_module("compute_cue", ROOT / "src" / "45_compute_factual_cue_metrics.py")
    branch = "llama33_70b"
    mod.BRANCHES[branch] = {"input": ITEM_TABLE, "output": CUE_METRICS}
    metrics = mod.run_branch(branch)
    if len(metrics) != 50:
        raise SystemExit("Cue metrics did not produce 50 rows.")


def compute_correctness() -> None:
    mod = load_module("correctness", ROOT / "src" / "46_join_factual_correctness.py")
    branch = "llama33_70b"
    mod.BRANCHES[branch] = {
        "metrics": CUE_METRICS,
        "original": WORKING_ORIGINAL,
        "paraphrase": WORKING_PARAPHRASE,
        "output": CORRECTNESS,
        "summary": CORRECTNESS_SUMMARY,
    }
    result = mod.run_branch(branch)
    if len(result) != 50:
        raise SystemExit("Correctness table did not produce 50 rows.")


def compute_text_features() -> None:
    mod = load_module("text_features", ROOT / "src" / "54_analyze_factual_text_feature_drivers.py")
    branch = "llama33_70b"
    mod.BRANCHES[branch] = {
        "label": "Llama 3.3 70B",
        "dir": INTERMEDIATE,
        "item": ITEM_TABLE,
        "cue": CUE_METRICS,
        "correctness": CORRECTNESS,
        "original": WORKING_ORIGINAL,
        "fixed": WORKING_PARAPHRASE,
    }
    paths = mod.run(branch)
    generated = {
        "base": paths["base"],
        "correlations": paths["correlations"],
        "regressions": paths["regressions"],
        "summary": paths["summary"],
    }
    generated["base"].replace(TEXT_BASE)
    generated["correlations"].replace(TEXT_CORR)
    generated["regressions"].replace(TEXT_REG)
    generated["summary"].replace(TEXT_SUMMARY)


def load_cross_model_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = {
        "GPT/main": {"base": GPT_BASE, "corr": GPT_CORR},
        "Llama 3.3 70B": {"base": TEXT_BASE, "corr": TEXT_CORR},
        "Qwen": {"base": QWEN_BASE, "corr": QWEN_CORR},
    }
    bases = []
    corrs = []
    for model, paths in files.items():
        base = pd.read_csv(paths["base"])
        corr = pd.read_csv(paths["corr"])
        base["model"] = model
        corr["model"] = model
        bases.append(base)
        corrs.append(corr)
    return pd.concat(bases, ignore_index=True), pd.concat(corrs, ignore_index=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    matrix = (
        corr[corr["feature"].isin(KEY_FEATURES)]
        .pivot(index="feature", columns="model", values="spearman_rho")
        .loc[KEY_FEATURES, MODELS]
    )
    labels = [FEATURE_LABELS.get(feature, feature) for feature in matrix.index]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Spearman correlation with factual QA paraphrase drift")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            color = "white" if abs(value) > 0.45 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    save_fig(OUT / "fig1_cross_model_feature_correlation_heatmap.png")


def plot_output_vs_prompt_strength(corr: pd.DataFrame) -> None:
    summary = (
        corr.groupby(["model", "feature_family"])["spearman_rho"]
        .apply(lambda s: s.abs().mean())
        .reset_index(name="mean_abs_rho")
    )
    families = ["output", "prompt"]
    x = range(len(MODELS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5))
    for offset, family in zip([-width / 2, width / 2], families):
        values = [
            float(summary[(summary["model"] == model) & (summary["feature_family"] == family)]["mean_abs_rho"].iloc[0])
            for model in MODELS
        ]
        bars = ax.bar([pos + offset for pos in x], values, width=width, label=f"{family}-side")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Mean absolute Spearman rho")
    ax.set_ylim(0, 0.62)
    ax.set_title("Output-side features are stronger than prompt-side features")
    ax.legend(frameon=False)
    save_fig(OUT / "fig2_output_vs_prompt_feature_strength.png")


def add_regression_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> tuple[float, float]:
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    slope, intercept, _, _, _ = stats.linregress(sub["x"], sub["y"])
    x_min, x_max = sub["x"].min(), sub["x"].max()
    ax.plot([x_min, x_max], [intercept + slope * x_min, intercept + slope * x_max], color=color, linewidth=2)
    rho, p = stats.spearmanr(sub["x"], sub["y"])
    return float(rho), float(p)


def scatter_grid(base: pd.DataFrame, feature: str, filename: str, title: str, x_label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = base[base["model"] == model]
        color = MODEL_COLORS[model]
        ax.scatter(
            sub[feature],
            sub["noise_corrected_drift"],
            s=34,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )
        rho, p = add_regression_line(ax, sub[feature], sub["noise_corrected_drift"], color)
        ax.set_title(f"{model}\nrho={rho:.2f}, p={p:.3g}", fontsize=10)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Noise-corrected drift")
    fig.suptitle(title, fontsize=13)
    save_fig(OUT / filename)


def plot_summary_table(corr: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        model_base = base[base["model"] == model]
        model_corr = corr[corr["model"] == model].set_index("feature")
        rows.append(
            {
                "model": model,
                "mean_ncp": model_base["noise_corrected_drift"].mean(),
                "mean_output_length_delta": model_base["output_length_delta_tokens"].mean(),
                "mean_output_length_ratio": model_base["output_length_ratio"].mean(),
                "mean_output_edit_rho": model_corr.loc["mean_output_token_edit_distance_norm", "spearman_rho"],
                "length_delta_rho": model_corr.loc["output_length_delta_tokens", "spearman_rho"],
                "f1_delta_rho": model_corr.loc["factual_score_delta", "spearman_rho"],
                "containment_delta_rho": model_corr.loc["containment_rate_delta", "spearman_rho"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "cross_model_text_feature_summary.csv", index=False)
    return summary


def write_readme(summary: pd.DataFrame) -> None:
    lines = [
        "# Token Analysis Figures: Llama 3.3 70B Refresh",
        "",
        "These figures refresh the fixed factual QA paraphrase text-feature driver analysis using the new Llama 3.3 70B n50 data.",
        "",
        "## Inputs",
        "",
        "- GPT/main: existing fixed factual QA text-feature tables in `outputs/`.",
        "- Llama 3.3 70B: recomputed from `llama/llama33_70b_instruct_turbo_150case/outputs/*_n50_four_task.csv`.",
        "- Qwen: existing fixed factual QA text-feature tables in `qwen/outputs/`.",
        "",
        "## Figures",
        "",
        "1. `fig1_cross_model_feature_correlation_heatmap.png`",
        "2. `fig2_output_vs_prompt_feature_strength.png`",
        "3. `fig3_output_edit_distance_vs_drift.png`",
        "4. `fig4_output_length_delta_vs_drift.png`",
        "5. `fig5_reference_f1_delta_vs_drift.png`",
        "6. `fig6_containment_delta_vs_drift.png`",
        "",
        "## Cross-model summary",
        "",
        "| Model | Mean NCP | Output length delta | Output length ratio | Output edit rho | Length rho | F1 delta rho | Containment rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {model} | {mean_ncp:.4f} | {mean_output_length_delta:.3f} | {mean_output_length_ratio:.3f} | "
            "{mean_output_edit_rho:.3f} | {length_delta_rho:.3f} | {f1_delta_rho:.3f} | {containment_delta_rho:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- New Llama 3.3 70B base table rows: 50.",
            "- New Llama 3.3 70B original factual QA rows: 250.",
            "- New Llama 3.3 70B factual QA paraphrasing rows: 250.",
            "- Old `token_analysis_figures/fig1` to `fig6` were not overwritten.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base, corr = load_cross_model_data()
    plot_correlation_heatmap(corr)
    plot_output_vs_prompt_strength(corr)
    scatter_grid(
        base,
        "mean_output_token_edit_distance_norm",
        "fig3_output_edit_distance_vs_drift.png",
        "Output token edit distance vs drift",
        "Mean output token edit distance",
    )
    scatter_grid(
        base,
        "output_length_delta_tokens",
        "fig4_output_length_delta_vs_drift.png",
        "Output length expansion vs drift",
        "Output length delta (tokens)",
    )
    scatter_grid(
        base,
        "factual_score_delta",
        "fig5_reference_f1_delta_vs_drift.png",
        "Reference token-F1 delta vs drift",
        "Reference token-F1 delta",
    )
    scatter_grid(
        base,
        "containment_rate_delta",
        "fig6_containment_delta_vs_drift.png",
        "Reference containment delta vs drift",
        "Reference containment delta",
    )
    summary = plot_summary_table(corr, base)
    write_readme(summary)


def validate_outputs() -> None:
    expected_pngs = [
        "fig1_cross_model_feature_correlation_heatmap.png",
        "fig2_output_vs_prompt_feature_strength.png",
        "fig3_output_edit_distance_vs_drift.png",
        "fig4_output_length_delta_vs_drift.png",
        "fig5_reference_f1_delta_vs_drift.png",
        "fig6_containment_delta_vs_drift.png",
    ]
    missing = [name for name in expected_pngs if not (OUT / name).exists() or (OUT / name).stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing or empty figure files: {missing}")
    base = pd.read_csv(TEXT_BASE)
    corr = pd.read_csv(TEXT_CORR)
    if len(base) != 50 or base["item_id"].nunique() != 50:
        raise SystemExit("New Llama 3.3 70B text feature base is not 50 rows/items.")
    missing_features = sorted(set(KEY_FEATURES) - set(corr["feature"]))
    if missing_features:
        raise SystemExit(f"Missing key features in Llama 3.3 70B correlations: {missing_features}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    extract_working_generation_files()
    recompute_sbert_cell()
    build_item_table()
    compute_cue_metrics()
    compute_correctness()
    compute_text_features()
    plot_all()
    validate_outputs()
    print(f"Wrote refreshed figures to {OUT}")


if __name__ == "__main__":
    main()
