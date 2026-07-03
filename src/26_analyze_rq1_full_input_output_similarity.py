"""Full analysis: input similarity vs output similarity for formal RQ1.

This script uses all 200 item-level perturbation rows from the formal RQ1 run.
It does not call the OpenAI API.
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

PERTURBED_PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts.csv"
EFFECTS = ROOT / "outputs" / "sbert_rq1_formal_perturbation_effects_by_item.csv"

ROWS = ROOT / "outputs" / "rq1_full_input_output_similarity_rows.csv"
CORRELATIONS = ROOT / "outputs" / "rq1_full_input_output_similarity_correlations.csv"
FIGURE = ROOT / "figures" / "attempt" / "rq1_full_input_output_similarity_scatter.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
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


def add_input_similarity() -> list[dict]:
    effects = read_csv(EFFECTS)
    prompt_lookup = {
        (row["item_id"], row["perturbation_type"]): row
        for row in read_csv(PERTURBED_PROMPTS)
    }
    sim_model = SimilarityModel()
    rows = []

    for row in effects:
        prompt_row = prompt_lookup[(row["item_id"], row["perturbation_type"])]
        input_similarity = sim_model.similarity(
            prompt_row["original_prompt"], prompt_row["perturbed_prompt"]
        )
        output_similarity = float(row["perturbation_similarity"])
        rows.append(
            {
                "item_id": row["item_id"],
                "task_type": row["task_type"],
                "perturbation_type": row["perturbation_type"],
                "input_similarity": round(input_similarity, 6),
                "input_drift": round(1 - input_similarity, 6),
                "output_similarity": round(output_similarity, 6),
                "output_drift": round(1 - output_similarity, 6),
                "baseline_similarity": row["baseline_similarity"],
                "noise_corrected_drift": row["noise_corrected_drift"],
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

    for perturbation_type in sorted({row["perturbation_type"] for row in rows}):
        perturbation_rows = [
            row for row in rows if row["perturbation_type"] == perturbation_type
        ]
        correlation_rows.append(
            correlation_row(f"perturbation={perturbation_type}", perturbation_rows)
        )

    for task_type in sorted({row["task_type"] for row in rows}):
        for perturbation_type in sorted({row["perturbation_type"] for row in rows}):
            subgroup = [
                row
                for row in rows
                if row["task_type"] == task_type
                and row["perturbation_type"] == perturbation_type
            ]
            correlation_rows.append(
                correlation_row(
                    f"task={task_type};perturbation={perturbation_type}",
                    subgroup,
                )
            )

    write_csv(CORRELATIONS, correlation_rows)


def plot_rows(rows: list[dict]) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["input_similarity"] = df["input_similarity"].astype(float)
    df["output_similarity"] = df["output_similarity"].astype(float)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="input_similarity",
        y="output_similarity",
        hue="perturbation_type",
        style="task_type",
        s=55,
        alpha=0.85,
    )
    sns.regplot(
        data=df,
        x="input_similarity",
        y="output_similarity",
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.2, "linestyle": "--"},
    )
    plt.title("Formal RQ1: Input Similarity vs Output Similarity")
    plt.xlabel("Input similarity: original prompt vs perturbed prompt")
    plt.ylabel("Output similarity: original outputs vs perturbed outputs")
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    rows = add_input_similarity()
    write_csv(ROWS, rows)
    write_correlations(rows)
    plot_rows(rows)

    print(f"Wrote {ROWS}")
    print(f"Wrote {CORRELATIONS}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
