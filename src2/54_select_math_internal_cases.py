"""Select math paraphrasing cases for internal drift analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts" / "rq1_formal_perturbed_prompts_n50_math_reasoning_fixed.csv"

BRANCHES = {
    "gpt": {
        "label": "GPT/main",
        "driver": ROOT / "outputs" / "math_fixed_paraphrase_driver_by_item.csv",
        "original": ROOT / "outputs" / "rq1_formal_original_generations_n50_math_reasoning.csv",
        "paraphrase": ROOT
        / "outputs"
        / "rq1_formal_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
    },
    "qwen": {
        "label": "Qwen",
        "driver": ROOT / "qwen" / "outputs" / "math_fixed_paraphrase_driver_by_item.csv",
        "original": ROOT / "qwen" / "outputs" / "rq1_qwen_original_generations_n50_math_reasoning.csv",
        "paraphrase": ROOT
        / "qwen"
        / "outputs"
        / "rq1_qwen_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
    },
    "llama": {
        "label": "Llama",
        "driver": ROOT / "llama" / "outputs" / "math_fixed_paraphrase_driver_by_item.csv",
        "original": ROOT / "llama" / "outputs" / "rq1_llama_original_generations_n50_math_reasoning.csv",
        "paraphrase": ROOT
        / "llama"
        / "outputs"
        / "rq1_llama_perturbed_generations_n50_math_reasoning_paraphrasing_fixed.csv",
    },
}

OUT_CSV = ROOT / "outputs" / "math_internal_case_selection.csv"
OUT_MD = ROOT / "outputs" / "math_internal_case_selection.md"


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def first_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, path, ["item_id", "sample_id", "output_text"])
    df = df.sort_values(["item_id", "sample_id"])
    return df.groupby("item_id", as_index=False).first()[["item_id", "output_text"]]


def load_prompts() -> pd.DataFrame:
    df = pd.read_csv(PROMPTS)
    require_columns(df, PROMPTS, ["item_id", "perturbation_type", "original_prompt", "perturbed_prompt"])
    df = df[df["perturbation_type"].eq("paraphrasing")].copy()
    df = df.drop_duplicates("item_id")
    return df[["item_id", "original_prompt", "perturbed_prompt"]]


def load_branch(key: str, spec: dict[str, Path | str]) -> pd.DataFrame:
    driver_path = Path(spec["driver"])
    driver = pd.read_csv(driver_path)
    require_columns(
        driver,
        driver_path,
        [
            "item_id",
            "noise_corrected_drift",
            "problem_content_recall",
            "math_token_recall",
            "number_recall",
            "output_length_delta_tokens",
            "answer_token_f1_delta",
            "answer_containment_delta",
        ],
    )
    driver = driver[driver["perturbation_type"].eq("paraphrasing")].copy()
    driver[f"{key}_ncp"] = driver["noise_corrected_drift"]
    driver[f"{key}_rank_desc"] = driver[f"{key}_ncp"].rank(method="first", ascending=False).astype(int)
    driver[f"{key}_high20"] = driver[f"{key}_rank_desc"].le(20)
    driver[f"{key}_low10"] = driver[f"{key}_rank_desc"].ge(len(driver) - 9)

    keep = [
        "item_id",
        f"{key}_ncp",
        f"{key}_rank_desc",
        f"{key}_high20",
        f"{key}_low10",
        "problem_content_recall",
        "math_token_recall",
        "number_recall",
        "output_length_delta_tokens",
        "answer_token_f1_delta",
        "answer_containment_delta",
    ]
    rename = {
        "problem_content_recall": f"{key}_problem_content_recall",
        "math_token_recall": f"{key}_math_token_recall",
        "number_recall": f"{key}_number_recall",
        "output_length_delta_tokens": f"{key}_output_length_delta_tokens",
        "answer_token_f1_delta": f"{key}_answer_token_f1_delta",
        "answer_containment_delta": f"{key}_answer_containment_delta",
    }
    branch = driver[keep].rename(columns=rename)

    original = first_output(Path(spec["original"])).rename(
        columns={"output_text": f"{key}_original_output_sample1"}
    )
    paraphrase = first_output(Path(spec["paraphrase"])).rename(
        columns={"output_text": f"{key}_paraphrase_output_sample1"}
    )
    branch = branch.merge(original, on="item_id", how="left").merge(paraphrase, on="item_id", how="left")
    return branch


def add_selection_groups(df: pd.DataFrame) -> pd.DataFrame:
    ncp_cols = [f"{key}_ncp" for key in BRANCHES]
    rank_cols = [f"{key}_rank_desc" for key in BRANCHES]
    high_cols = [f"{key}_high20" for key in BRANCHES]
    low_cols = [f"{key}_low10" for key in BRANCHES]

    df["mean_ncp"] = df[ncp_cols].mean(axis=1)
    df["mean_rank"] = df[rank_cols].mean(axis=1)
    df["high20_branch_count"] = df[high_cols].sum(axis=1).astype(int)
    df["low10_branch_count"] = df[low_cols].sum(axis=1).astype(int)

    medians = {key: df[f"{key}_ncp"].median() for key in BRANCHES}
    df["above_all_branch_medians"] = True
    for key, median in medians.items():
        df["above_all_branch_medians"] &= df[f"{key}_ncp"].ge(median)

    strict_shared = df[df["high20_branch_count"].eq(3)].sort_values(["mean_rank", "mean_ncp"], ascending=[True, False])
    if len(strict_shared) >= 5:
        shared_ids = set(strict_shared.head(10)["item_id"])
        shared_rule = "strict intersection of top-20 high-drift cases across GPT/Qwen/Llama"
    else:
        pooled = df[df["above_all_branch_medians"]].sort_values(
            ["high20_branch_count", "mean_rank", "mean_ncp"], ascending=[False, True, False]
        )
        shared_ids = set(pooled.head(10)["item_id"])
        shared_rule = (
            "strict top-20 intersection had fewer than 5 cases, so shared cases use all-branch "
            "above-median drift plus best mean rank"
        )

    groups_by_id: dict[str, list[str]] = {item_id: [] for item_id in df["item_id"]}
    for key in BRANCHES:
        high_ids = set(df[df[f"{key}_high20"]]["item_id"])
        low_ids = set(df[df[f"{key}_low10"]]["item_id"])
        for item_id in high_ids:
            groups_by_id[item_id].append(f"{key}_high20")
        for item_id in low_ids:
            groups_by_id[item_id].append(f"{key}_low10")
    for item_id in shared_ids:
        groups_by_id[item_id].append("cross_model_shared_high")

    df["selection_group"] = df["item_id"].map(lambda item_id: ";".join(groups_by_id[item_id]))
    selected = df[df["selection_group"].ne("")].copy()
    selected["shared_selection_rule"] = shared_rule
    selected = selected.sort_values(
        ["selection_group", "mean_rank", "mean_ncp"], ascending=[True, True, False]
    )
    return selected


def write_summary(selected: pd.DataFrame, all_cases: pd.DataFrame) -> None:
    lines = [
        "# Math internal case selection",
        "",
        "## Validation",
        "",
        f"- Total math paraphrasing items available: {len(all_cases)}",
        f"- Selected rows: {len(selected)}",
        f"- Cross-model shared high rows: {(selected['selection_group'].str.contains('cross_model_shared_high')).sum()}",
        f"- Shared selection rule: {selected['shared_selection_rule'].iloc[0] if len(selected) else 'n/a'}",
        "",
        "## Per-branch coverage",
        "",
    ]
    for key, spec in BRANCHES.items():
        lines.append(f"- {spec['label']} high20 selected: {(selected[f'{key}_high20']).sum()}")
        lines.append(f"- {spec['label']} low10 selected: {(selected[f'{key}_low10']).sum()}")

    required_prompt_missing = selected[["original_prompt", "perturbed_prompt"]].isna().any(axis=1).sum()
    output_cols = [column for column in selected.columns if column.endswith("_output_sample1")]
    output_missing = selected[output_cols].isna().any(axis=1).sum()
    lines.extend(
        [
            "",
            "## Completeness checks",
            "",
            f"- Rows missing original/paraphrased prompt: {required_prompt_missing}",
            f"- Rows missing at least one branch output sample: {output_missing}",
            "",
            "## Top cross-model shared candidates",
            "",
        ]
    )

    display_cols = ["item_id", "mean_ncp", "mean_rank", "high20_branch_count", "selection_group"]
    shared = selected[selected["selection_group"].str.contains("cross_model_shared_high")].copy()
    for _, row in shared.sort_values(["mean_rank", "mean_ncp"], ascending=[True, False]).head(10)[display_cols].iterrows():
        lines.append(
            f"- {row['item_id']}: mean_ncp={row['mean_ncp']:.6f}, "
            f"mean_rank={row['mean_rank']:.2f}, high20_branch_count={int(row['high20_branch_count'])}"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    prompts = load_prompts()
    merged = prompts.copy()
    for key, spec in BRANCHES.items():
        merged = merged.merge(load_branch(key, spec), on="item_id", how="inner")

    selected = add_selection_groups(merged)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT_CSV, index=False)
    write_summary(selected, merged)
    print(f"wrote {OUT_CSV} rows={len(selected)}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
