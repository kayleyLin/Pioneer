"""Create heatmaps for expanded RQ1 n=50 perturbation results (Llama outputs)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / "llama"
FIGURES = LLAMA / "figures"
CORRECTED = LLAMA / "outputs" / "sbert_rq1_n50_heatmap_noise_corrected_drift.csv"
UNCORRECTED = LLAMA / "outputs" / "sbert_rq1_n50_uncorrected_heatmap_drift.csv"

TASK_ORDER = [
    "code_generation",
    "factual_qa",
    "math_reasoning",
    "open_ended_writing",
]
PERTURBATION_ORDER = [
    "paraphrasing",
    "reordering",
    "formatting_changes",
    "context_injection",
    "surface_noise",
]


def load_heatmap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("perturbation_type")
    return df.reindex(index=PERTURBATION_ORDER, columns=TASK_ORDER).astype(float)


def save_heatmap(data: pd.DataFrame, path: Path, title: str, label: str) -> None:
    plt.figure(figsize=(9.2, 5.4))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": label},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Task type")
    ax.set_ylabel("Perturbation type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    corrected = load_heatmap(CORRECTED)
    uncorrected = load_heatmap(UNCORRECTED)

    save_heatmap(
        uncorrected,
        FIGURES / "rq1_n50_uncorrected_drift_heatmap.png",
        "RQ1 n=50 (Llama): Uncorrected Semantic Drift",
        "Mean uncorrected drift",
    )
    save_heatmap(
        corrected,
        FIGURES / "rq1_n50_noise_corrected_drift_heatmap.png",
        "RQ1 n=50 (Llama): Noise-Corrected Semantic Drift",
        "Mean noise-corrected drift",
    )
    print(FIGURES / "rq1_n50_uncorrected_drift_heatmap.png")
    print(FIGURES / "rq1_n50_noise_corrected_drift_heatmap.png")


if __name__ == "__main__":
    main()

