"""Analyze input similarity vs output similarity for surface-noise intensity data.

This script does not call any API. It combines:

    prompts/rq1_surface_noise_intensity_prompts.csv
    outputs/sbert_rq1_surface_noise_intensity_by_item.csv

and writes input-output similarity rows, correlations, and scatter plots.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PERTURBED_PROMPTS = ROOT / "prompts" / "rq1_surface_noise_intensity_prompts.csv"
EFFECTS = ROOT / "outputs" / "sbert_rq1_surface_noise_intensity_by_item.csv"

ROWS = ROOT / "outputs" / "rq1_surface_noise_input_output_similarity_rows.csv"
CORRELATIONS = ROOT / "outputs" / "rq1_surface_noise_input_output_similarity_correlations.csv"
OVERALL_FIGURE = (
    ROOT / "figures" / "attempt" / "rq1_surface_noise_input_output_similarity_scatter.png"
)
TASK_FIGURE = (
    ROOT / "figures" / "attempt" / "rq1_surface_noise_input_output_similarity_by_task.png"
)

INTENSITY_LABELS = {
    "surface_noise_low": "low",
    "surface_noise_medium": "medium",
    "surface_noise_high": "high",
}
INTENSITY_ORDER = ["low", "medium", "high"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class SimilarityModel:
    def __init__(self) -> None:
        print(f"Loading Sentence-BERT model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.cache = {}

    def encode(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self.cache[text]

    def similarity(self, text_a: str, text_b: str) -> float:
        return float(cos_sim(self.encode(text_a), self.encode(text_b))[0][0])


def build_rows() -> list[dict]:
    prompt_lookup = {
        (row["item_id"], row["perturbation_type"]): row
        for row in read_csv(PERTURBED_PROMPTS)
    }
    effects = read_csv(EFFECTS)
    sim_model = SimilarityModel()
    rows = []

    for effect in effects:
        key = (effect["item_id"], effect["perturbation_type"])
        prompt_row = prompt_lookup[key]
        input_similarity = sim_model.similarity(
            prompt_row["original_prompt"], prompt_row["perturbed_prompt"]
        )
        output_similarity = float(effect["perturbation_similarity"])
        rows.append(
            {
                "item_id": effect["item_id"],
                "task_type": effect["task_type"],
                "perturbation_type": effect["perturbation_type"],
                "perturbation_intensity": INTENSITY_LABELS[effect["perturbation_type"]],
                "surface_noise_error_count": prompt_row["surface_noise_error_count"],
                "surface_noise_eligible_token_count": prompt_row[
                    "surface_noise_eligible_token_count"
                ],
                "input_similarity": round(input_similarity, 6),
                "input_drift": round(1 - input_similarity, 6),
                "output_similarity": round(output_similarity, 6),
                "output_drift": round(1 - output_similarity, 6),
                "baseline_similarity": effect["baseline_similarity"],
                "noise_corrected_drift": effect["noise_corrected_drift"],
                "similarity_metric": MODEL_NAME,
            }
        )
    return rows


def correlation_row(label: str, rows: list[dict]) -> dict:
    x = [float(row["input_similarity"]) for row in rows]
    y = [float(row["output_similarity"]) for row in rows]
    if len(rows) < 3:
        return {
            "group": label,
            "n": len(rows),
            "pearson_r": "",
            "pearson_p": "",
            "spearman_rho": "",
            "spearman_p": "",
        }

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "group": label,
        "n": len(rows),
        "pearson_r": round(float(pearson.statistic), 6),
        "pearson_p": round(float(pearson.pvalue), 6),
        "spearman_rho": round(float(spearman.statistic), 6),
        "spearman_p": round(float(spearman.pvalue), 6),
    }


def write_correlations(rows: list[dict]) -> None:
    correlation_rows = [correlation_row("overall", rows)]

    for task_type in sorted({row["task_type"] for row in rows}):
        task_rows = [row for row in rows if row["task_type"] == task_type]
        correlation_rows.append(correlation_row(f"task={task_type}", task_rows))

    for intensity in INTENSITY_ORDER:
        intensity_rows = [
            row for row in rows if row["perturbation_intensity"] == intensity
        ]
        correlation_rows.append(correlation_row(f"intensity={intensity}", intensity_rows))

    for task_type in sorted({row["task_type"] for row in rows}):
        for intensity in INTENSITY_ORDER:
            subgroup = [
                row
                for row in rows
                if row["task_type"] == task_type
                and row["perturbation_intensity"] == intensity
            ]
            correlation_rows.append(
                correlation_row(f"task={task_type};intensity={intensity}", subgroup)
            )

    write_csv(CORRELATIONS, correlation_rows)


def plot_overall(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["input_similarity"] = df["input_similarity"].astype(float)
    df["output_similarity"] = df["output_similarity"].astype(float)
    df["perturbation_intensity"] = pd.Categorical(
        df["perturbation_intensity"], categories=INTENSITY_ORDER, ordered=True
    )

    OVERALL_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="input_similarity",
        y="output_similarity",
        hue="perturbation_intensity",
        style="task_type",
        hue_order=INTENSITY_ORDER,
        s=65,
        alpha=0.82,
    )
    sns.regplot(
        data=df,
        x="input_similarity",
        y="output_similarity",
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.2, "linestyle": "--"},
    )
    plt.title("Surface Noise: Input Similarity vs Output Similarity")
    plt.xlabel("Input similarity: original prompt vs perturbed prompt")
    plt.ylabel("Output similarity: original outputs vs perturbed outputs")
    plt.tight_layout()
    plt.savefig(OVERALL_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()


def plot_by_task(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["input_similarity"] = df["input_similarity"].astype(float)
    df["output_similarity"] = df["output_similarity"].astype(float)
    df["perturbation_intensity"] = pd.Categorical(
        df["perturbation_intensity"], categories=INTENSITY_ORDER, ordered=True
    )

    grid = sns.lmplot(
        data=df,
        x="input_similarity",
        y="output_similarity",
        hue="perturbation_intensity",
        col="task_type",
        col_wrap=2,
        hue_order=INTENSITY_ORDER,
        height=4,
        aspect=1.1,
        scatter_kws={"s": 45, "alpha": 0.8},
        line_kws={"linewidth": 1.1},
        ci=None,
    )
    grid.fig.suptitle(
        "Surface Noise: Input Similarity vs Output Similarity By Task",
        y=1.03,
    )
    grid.set_axis_labels(
        "Input similarity: original vs perturbed prompt",
        "Output similarity: original vs perturbed outputs",
    )
    TASK_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(TASK_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    rows = build_rows()
    write_csv(ROWS, rows)
    write_correlations(rows)
    plot_overall(rows)
    plot_by_task(rows)

    print(f"Wrote {ROWS}")
    print(f"Wrote {CORRELATIONS}")
    print(f"Wrote {OVERALL_FIGURE}")
    print(f"Wrote {TASK_FIGURE}")


if __name__ == "__main__":
    main()
