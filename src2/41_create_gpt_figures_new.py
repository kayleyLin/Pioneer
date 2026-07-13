"""Create GPT/main fig1-fig4 in figures_new using the unified figure style."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

import matplotlib


matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
FIGURES_NEW = ROOT / "figures_new"
RQ2_OUTPUTS = ROOT / "rq2_outputs"
RQ2_AVAILABLE = ROOT / "rq2" / "outputs" / "rq2_formal_available_drift_performance_by_item.csv"


def load_unified_figures_module():
    path = ROOT / "src" / "40_create_unified_model_figures.py"
    spec = importlib.util.spec_from_file_location("unified_model_figures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_fig4_source() -> Path:
    source = RQ2_OUTPUTS / "fig4_similarity_correctness_data.csv"
    if source.exists():
        return source

    if not RQ2_AVAILABLE.exists():
        raise FileNotFoundError(f"Missing GPT RQ2 source data: {RQ2_AVAILABLE}")

    df = pd.read_csv(RQ2_AVAILABLE)
    df["original_performance"] = pd.to_numeric(df["original_performance"], errors="coerce")
    df["perturbed_performance"] = pd.to_numeric(df["perturbed_performance"], errors="coerce")
    df["correctness_retention"] = (
        df["perturbed_performance"] / df["original_performance"]
    ).clip(upper=1)
    df.loc[df["original_performance"] <= 0, "correctness_retention"] = pd.NA
    df["similarity"] = pd.to_numeric(df["perturbation_similarity"], errors="coerce")

    out = df[
        [
            "item_id",
            "task_type",
            "perturbation_type",
            "similarity",
            "perturbed_performance",
            "original_performance",
            "correctness_retention",
            "absolute_performance_change",
            "pdr",
            "similarity_metric",
        ]
    ].copy()

    RQ2_OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(source, index=False)
    return source


def copy_unprefixed_outputs() -> None:
    mapping = {
        "gpt_fig1_noise_baseline.png": "fig1_noise_baseline.png",
        "gpt_fig2_tukey_hsd.png": "fig2_tukey_hsd.png",
        "gpt_fig3_drift_heatmap.png": "fig3_drift_heatmap.png",
        "gpt_fig4_similarity_performance_retention.png": "fig4_similarity_performance_retention.png",
    }
    for src_name, dst_name in mapping.items():
        src = FIGURES_NEW / src_name
        if not src.exists():
            raise FileNotFoundError(f"Expected generated figure not found: {src}")
        shutil.copy2(src, FIGURES_NEW / dst_name)
        src.unlink()


def main() -> None:
    module = load_unified_figures_module()
    module.MODEL_COMPARISON_FIGURES = FIGURES_NEW

    build_fig4_source()
    FIGURES_NEW.mkdir(parents=True, exist_ok=True)
    module.configure_style()

    model = {
        "key": "gpt",
        "label": "GPT-4o mini",
        "outputs": ROOT / "outputs",
        "rq2_outputs": RQ2_OUTPUTS,
        "fig4_real": True,
    }
    module.figure_1(model)
    module.figure_2(model)
    module.figure_3(model)
    module.figure_4(model)
    copy_unprefixed_outputs()


if __name__ == "__main__":
    main()
