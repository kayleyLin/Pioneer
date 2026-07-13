"""Run the expanded RQ2 analysis on n50 + add100 GPT/main outputs.

The output is a self-contained Markdown report at ``RQ2_150.md`` plus
supporting CSV tables in ``rq2_150_outputs/``.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import statsmodels.formula.api as smf
from scipy import stats

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "rq2_150_outputs"
REPORT = ROOT / "RQ2_150.md"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RQ2_TASKS = {"factual_qa", "math_reasoning", "code_generation"}
PERTURBATIONS = [
    "context_injection",
    "formatting_changes",
    "paraphrasing",
    "reordering",
    "surface_noise",
]

ORIGINAL_GENERATION_FILES = [
    ROOT / "outputs" / "rq1_formal_original_generations_n50_factual_qa.csv",
    ROOT / "outputs" / "rq1_formal_original_generations_n50_math_reasoning.csv",
    ROOT / "outputs" / "rq1_formal_original_generations_n50_code_generation.csv",
    ROOT / "outputs" / "rq1_formal_original_generations_add100_three_task.csv",
]
PERTURBED_GENERATION_FILES = [
    ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_factual_qa_fixed.csv",
    ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_math_reasoning_fixed.csv",
    ROOT / "outputs" / "rq1_formal_perturbed_generations_n50_code_generation.csv",
    ROOT / "outputs" / "rq1_formal_perturbed_generations_add100_three_task.csv",
]
PROMPT_FILES = [
    ROOT / "prompts" / "rq1_sampled_original_prompts_n50_factual_qa.csv",
    ROOT / "prompts" / "rq1_sampled_original_prompts_n50_math_reasoning.csv",
    ROOT / "prompts" / "rq1_sampled_original_prompts_n50_code_generation.csv",
    ROOT / "prompts" / "rq1_sampled_original_prompts_add100_three_task.csv",
]


@dataclass
class EvaluationResult:
    extracted_answer: str
    is_correct: bool | None
    needs_manual_review: bool
    correctness_method: str
    notes: str
    performance_score: float | None = None
    factual_containment_match: bool | None = None
    factual_token_f1: float | None = None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str, remove_articles: bool = False) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\s+", " ", text).strip()
    if remove_articles:
        text = " ".join(word for word in text.split() if word not in {"a", "an", "the"})
    return text


def token_overlap_ratio(reference: str, output: str) -> float:
    ref_tokens = set(normalize_text(reference, remove_articles=True).split())
    out_tokens = set(normalize_text(output, remove_articles=True).split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & out_tokens) / len(ref_tokens)


def token_f1(reference: str, output: str) -> float:
    ref_tokens = normalize_text(reference, remove_articles=True).split()
    out_tokens = normalize_text(output, remove_articles=True).split()
    if not ref_tokens or not out_tokens:
        return 0.0
    common = Counter(ref_tokens) & Counter(out_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(out_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_factual_qa(output_text: str, reference_answer: str) -> EvaluationResult:
    normalized_ref = normalize_text(reference_answer, remove_articles=True)
    normalized_output = normalize_text(output_text, remove_articles=True)
    if not normalized_ref:
        return EvaluationResult("", False, False, "factual_qa_containment_token_f1", "missing reference answer", 0.0, False, 0.0)
    if normalized_ref in normalized_output:
        return EvaluationResult(reference_answer, None, False, "factual_qa_containment_token_f1", "reference answer contained", 1.0, True, 1.0)
    overlap = token_overlap_ratio(reference_answer, output_text)
    f1 = token_f1(reference_answer, output_text)
    return EvaluationResult(reference_answer, None, False, "factual_qa_containment_token_f1", f"token overlap={overlap:.2f}", f1, False, f1)


ANSWER_PHRASE_RE = re.compile(r"(?:final answer|answer|therefore|thus)\s*(?:is|=|:)?\s*([^\n.]+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?")


def extract_last_latex_command_argument(text: str, command: str) -> str:
    marker = f"\\{command}"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        brace_start = text.find("{", start + len(marker))
        if brace_start == -1:
            continue
        depth = 0
        for index in range(brace_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1 : index]
    return ""


def replace_latex_fractions(text: str) -> str:
    return re.sub(r"\\(?:dfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)


def clean_math_candidate(text: str) -> str:
    text = str(text).strip()
    text = text.replace("$", "").replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = replace_latex_fractions(text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace(",", "").strip(" .,:;")
    return text


def extract_math_answer(output_text: str) -> str:
    boxed = extract_last_latex_command_argument(output_text, "boxed")
    if boxed:
        return clean_math_candidate(boxed)
    phrase_matches = ANSWER_PHRASE_RE.findall(output_text)
    if phrase_matches:
        candidate = clean_math_candidate(phrase_matches[-1])
        numbers = NUMBER_RE.findall(candidate)
        return clean_math_candidate(numbers[-1]) if numbers else candidate
    numbers = NUMBER_RE.findall(output_text)
    return clean_math_candidate(numbers[-1]) if numbers else ""


def normalize_math_string(text: str) -> str:
    return re.sub(r"\s+", "", clean_math_candidate(text).lower())


def sympy_equivalent(left: str, right: str) -> bool | None:
    if sp is None:
        return None
    left = clean_math_candidate(left)
    right = clean_math_candidate(right)
    if not left or not right:
        return None
    try:
        return bool(sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0)
    except Exception:
        return None


def evaluate_math_reasoning(output_text: str, reference_answer: str) -> EvaluationResult:
    extracted = extract_math_answer(output_text)
    if not extracted:
        return EvaluationResult("", False, False, "math_final_answer_equivalence", "could not extract final answer", 0.0)
    if normalize_math_string(extracted) == normalize_math_string(reference_answer):
        return EvaluationResult(extracted, True, False, "math_final_answer_equivalence", "normalized answer matched", 1.0)
    symbolic = sympy_equivalent(extracted, reference_answer)
    if symbolic is True:
        return EvaluationResult(extracted, True, False, "math_final_answer_equivalence", "symbolic equivalence matched", 1.0)
    if symbolic is False:
        return EvaluationResult(extracted, False, False, "math_final_answer_equivalence", "symbolic equivalence failed", 0.0)
    return EvaluationResult(extracted, False, False, "math_final_answer_equivalence", "could not safely compare", 0.0)


def extract_python_code(output_text: str, entry_point: str = "", declaration: str = "") -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", output_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()
    lines = output_text.splitlines()
    start_index = None
    if entry_point:
        pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
        for index, line in enumerate(lines):
            if pattern.search(line):
                start_index = index
                break
    if start_index is None:
        for index, line in enumerate(lines):
            if re.match(r"^\s*(from\s+\S+\s+import\s+|import\s+|def\s+\w+\s*\()", line):
                start_index = index
                break
    code = "\n".join(lines[start_index:]).strip() if start_index is not None else output_text.strip()
    if entry_point and f"def {entry_point}" not in code and declaration:
        body = "\n".join(f"    {line}" if line.strip() else line for line in code.splitlines())
        code = f"{declaration.rstrip()}\n{body}"
    return code.strip()


def code_is_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def load_humanevalpack_tests() -> dict[str, dict[str, str]]:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "bigcode___humanevalpack" / "python"
    arrow_files = sorted(cache_root.rglob("humanevalpack-test.arrow")) if cache_root.exists() else []
    if arrow_files:
        table = ipc.open_stream(str(arrow_files[-1])).read_all()
        rows = table.to_pylist()
    else:
        from datasets import load_dataset

        rows = list(load_dataset("bigcode/humanevalpack", "python", split="test"))
    return {
        str(index): {
            "task_id": str(row.get("task_id", "")),
            "entry_point": str(row.get("entry_point", "")),
            "declaration": str(row.get("declaration", "")),
            "imports": str(row.get("import", "")),
            "test_setup": str(row.get("test_setup", "")),
            "test": str(row.get("test", "")),
        }
        for index, row in enumerate(rows)
    }


def evaluate_code_generation(output_text: str, test_info: dict[str, str] | None) -> EvaluationResult:
    if not test_info:
        return EvaluationResult("", None, True, "humanevalpack_functional_tests", "missing test metadata", None)
    entry_point = test_info.get("entry_point", "")
    code = extract_python_code(output_text, entry_point=entry_point, declaration=test_info.get("declaration", ""))
    if not code or not entry_point:
        return EvaluationResult(code, None, True, "humanevalpack_functional_tests", "missing code or entry point", None)
    if not code_is_parseable(code):
        return EvaluationResult(code, False, False, "humanevalpack_functional_tests", "syntax error", 0.0)
    runner = "\n\n".join(part for part in [test_info.get("imports", ""), test_info.get("test_setup", ""), code, test_info.get("test", "")] if part.strip())
    with tempfile.TemporaryDirectory(prefix="rq2_150_code_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner, encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=5, check=False)
        except subprocess.TimeoutExpired:
            return EvaluationResult(code, False, False, "humanevalpack_functional_tests", "execution timed out", 0.0)
    if completed.returncode == 0:
        return EvaluationResult(code, True, False, "humanevalpack_functional_tests", "passed tests", 1.0)
    error_tail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
    note = error_tail[0] if error_tail else f"test process returned {completed.returncode}"
    return EvaluationResult(code, False, False, "humanevalpack_functional_tests", note[:300], 0.0)


def bool_to_csv(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def evaluate_generation_rows(rows: list[dict[str, str]], prompt_by_item: dict[str, dict[str, str]], code_tests: dict[str, dict[str, str]], variant: str) -> list[dict[str, str]]:
    output_rows = []
    for index, row in enumerate(rows, 1):
        if index % 1000 == 0:
            print(f"Evaluated {index}/{len(rows)} {variant} rows")
        task_type = row["task_type"]
        prompt_meta = prompt_by_item.get(row["item_id"], {})
        if task_type == "factual_qa":
            result = evaluate_factual_qa(row.get("output_text", ""), prompt_meta.get("reference_answer", ""))
        elif task_type == "math_reasoning":
            result = evaluate_math_reasoning(row.get("output_text", ""), prompt_meta.get("reference_answer", ""))
        elif task_type == "code_generation":
            result = evaluate_code_generation(row.get("output_text", ""), code_tests.get(row.get("source_index", "")))
        else:
            continue
        output_rows.append(
            {
                "item_id": row.get("item_id", ""),
                "task_type": task_type,
                "dataset_name": row.get("dataset_name", ""),
                "source_index": row.get("source_index", ""),
                "source_id": prompt_meta.get("source_id", ""),
                "sample_id": row.get("sample_id", ""),
                "model_name": row.get("model_name", ""),
                "prompt_variant": variant,
                "perturbation_type": row.get("perturbation_type", ""),
                "reference_answer": prompt_meta.get("reference_answer", ""),
                "output_text": row.get("output_text", ""),
                "extracted_answer": result.extracted_answer,
                "performance_score": "" if result.performance_score is None else f"{result.performance_score:.6f}",
                "factual_containment_match": "" if result.factual_containment_match is None else ("true" if result.factual_containment_match else "false"),
                "factual_token_f1": "" if result.factual_token_f1 is None else f"{result.factual_token_f1:.6f}",
                "is_correct": bool_to_csv(result.is_correct),
                "needs_manual_review": "true" if result.needs_manual_review else "false",
                "correctness_method": result.correctness_method,
                "notes": result.notes,
            }
        )
    return output_rows


def load_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, str]]]:
    prompts = [row for path in PROMPT_FILES for row in read_csv_rows(path)]
    prompt_by_item = {row["item_id"]: row for row in prompts if row.get("task_type") in RQ2_TASKS}
    original_rows = [row for path in ORIGINAL_GENERATION_FILES for row in read_csv_rows(path) if row.get("task_type") in RQ2_TASKS]
    perturbed_rows = [row for path in PERTURBED_GENERATION_FILES for row in read_csv_rows(path) if row.get("task_type") in RQ2_TASKS]
    return original_rows, perturbed_rows, prompt_by_item


def build_performance_change(original_eval: list[dict[str, str]], perturbed_eval: list[dict[str, str]]) -> list[dict[str, str]]:
    original_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    perturbed_by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in original_eval:
        original_by_item[row["item_id"]].append(row)
    for row in perturbed_eval:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(row)

    rows = []
    for (item_id, perturbation_type), group in sorted(perturbed_by_group.items()):
        if item_id not in original_by_item:
            continue
        original_scores = [float(row["performance_score"]) for row in original_by_item[item_id] if row["performance_score"] != ""]
        perturbed_scores = [float(row["performance_score"]) for row in group if row["performance_score"] != ""]
        if not original_scores or not perturbed_scores:
            continue
        original_performance = mean(original_scores)
        perturbed_performance = mean(perturbed_scores)
        change = original_performance - perturbed_performance
        rows.append(
            {
                "item_id": item_id,
                "task_type": group[0]["task_type"],
                "perturbation_type": perturbation_type,
                "n_original_outputs": str(len(original_scores)),
                "n_perturbed_outputs": str(len(perturbed_scores)),
                "original_performance": f"{original_performance:.6f}",
                "perturbed_performance": f"{perturbed_performance:.6f}",
                "absolute_performance_change": f"{change:.6f}",
                "pdr": "" if original_performance == 0 else f"{change / original_performance:.6f}",
                "performance_dropped": "true" if perturbed_performance < original_performance else "false",
            }
        )
    return rows


def compute_drift_table(original_rows: list[dict[str, str]], perturbed_rows: list[dict[str, str]], performance_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    original_by_item: dict[str, list[str]] = defaultdict(list)
    perturbed_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in original_rows:
        original_by_item[row["item_id"]].append(row.get("output_text", ""))
    for row in perturbed_rows:
        perturbed_by_group[(row["item_id"], row["perturbation_type"])].append(row.get("output_text", ""))

    all_texts = sorted({text for texts in original_by_item.values() for text in texts} | {text for texts in perturbed_by_group.values() for text in texts})
    print(f"Encoding {len(all_texts)} unique outputs with {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    emb_by_text = dict(zip(all_texts, embeddings))

    def sim(a: str, b: str) -> float:
        return float(np.dot(emb_by_text[a], emb_by_text[b]))

    def within_similarity(outputs: list[str]) -> float:
        return mean(sim(a, b) for a, b in combinations(outputs, 2))

    def cross_similarity(originals: list[str], perturbed: list[str]) -> float:
        return mean(sim(a, b) for a, b in product(originals, perturbed))

    baseline_by_item = {item_id: within_similarity(outputs) for item_id, outputs in original_by_item.items()}
    perf_by_group = {(row["item_id"], row["perturbation_type"]): row for row in performance_rows}
    out_rows = []
    for key, pert_outputs in sorted(perturbed_by_group.items()):
        item_id, perturbation_type = key
        if item_id not in original_by_item or key not in perf_by_group:
            continue
        baseline = baseline_by_item[item_id]
        pert_sim = cross_similarity(original_by_item[item_id], pert_outputs)
        drift = baseline - pert_sim
        out_rows.append(
            {
                **perf_by_group[key],
                "baseline_similarity": f"{baseline:.6f}",
                "perturbation_similarity": f"{pert_sim:.6f}",
                "noise_corrected_drift": f"{drift:.6f}",
                "similarity_metric": MODEL_NAME,
            }
        )
    return out_rows


def holm_adjust(pvalues: list[float]) -> list[float]:
    valid = [(i, p) for i, p in enumerate(pvalues) if not math.isnan(p)]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted = [float("nan")] * len(pvalues)
    running = 0.0
    m = len(ordered)
    for rank, (index, pvalue) in enumerate(ordered):
        adj = (m - rank) * pvalue
        running = max(running, adj)
        adjusted[index] = min(running, 1.0)
    return adjusted


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    specs: list[tuple[str, list[str] | None]] = [
        ("overall", None),
        ("task", ["task_type"]),
        ("perturbation", ["perturbation_type"]),
        ("task_perturbation", ["task_type", "perturbation_type"]),
    ]
    rows = []
    for group_family, group_cols in specs:
        groups = [((None,), df)] if group_cols is None else list(df.groupby(group_cols, dropna=False))
        for keys, group in groups:
            if group_cols is None:
                labels = {"group_family": group_family, "group": "overall", "task_type": "", "perturbation_type": ""}
            else:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                labels = {"group_family": group_family, "group": " / ".join(map(str, keys)), "task_type": "", "perturbation_type": ""}
                labels.update(dict(zip(group_cols, keys)))
            valid = group[["noise_corrected_drift", "absolute_performance_change"]].dropna()
            row = {**labels, "n": len(valid)}
            if len(valid) >= 3 and valid["noise_corrected_drift"].nunique() > 1 and valid["absolute_performance_change"].nunique() > 1:
                pearson = stats.pearsonr(valid["noise_corrected_drift"], valid["absolute_performance_change"])
                spearman = stats.spearmanr(valid["noise_corrected_drift"], valid["absolute_performance_change"])
                row.update({"pearson_r": pearson.statistic, "pearson_p": pearson.pvalue, "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue})
            else:
                row.update({"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")})
            rows.append(row)
    corr = pd.DataFrame(rows)
    corr["spearman_holm_p"] = holm_adjust(corr["spearman_p"].tolist())
    return corr


def descriptive_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(cols, keys))
        row["n"] = len(group)
        row["mean_noise_corrected_drift"] = group["noise_corrected_drift"].mean()
        row["mean_absolute_performance_change"] = group["absolute_performance_change"].mean()
        row["mean_original_performance"] = group["original_performance"].mean()
        row["mean_perturbed_performance"] = group["perturbed_performance"].mean()
        row["drop_share"] = (group["performance_dropped"] == "true").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def factual_robustness(df: pd.DataFrame) -> dict[str, Any]:
    factual = df[df["task_type"] == "factual_qa"].copy()
    raw = stats.spearmanr(factual["noise_corrected_drift"], factual["absolute_performance_change"])
    model = smf.ols("absolute_performance_change ~ noise_corrected_drift + C(perturbation_type)", data=factual).fit()
    cluster = smf.ols("absolute_performance_change ~ noise_corrected_drift + C(perturbation_type)", data=factual).fit(
        cov_type="cluster",
        cov_kwds={"groups": factual["item_id"]},
    )
    coef = float(cluster.params["noise_corrected_drift"])
    se = float(cluster.bse["noise_corrected_drift"])
    t_stat = coef / se
    clusters = factual["item_id"].nunique()
    cluster_t_p = 2 * stats.t.sf(abs(t_stat), clusters - 1)

    centered = factual.copy()
    centered["drift_centered"] = centered["noise_corrected_drift"] - centered.groupby("perturbation_type")["noise_corrected_drift"].transform("mean")
    centered["change_centered"] = centered["absolute_performance_change"] - centered.groupby("perturbation_type")["absolute_performance_change"].transform("mean")
    centered_s = stats.spearmanr(centered["drift_centered"], centered["change_centered"])

    per_pert = []
    for pert, group in factual.groupby("perturbation_type"):
        result = stats.spearmanr(group["noise_corrected_drift"], group["absolute_performance_change"])
        per_pert.append({"perturbation_type": pert, "n": len(group), "spearman_rho": result.statistic, "spearman_p": result.pvalue})

    return {
        "raw_spearman_rho": raw.statistic,
        "raw_spearman_p": raw.pvalue,
        "ols_beta": float(model.params["noise_corrected_drift"]),
        "ols_p": float(model.pvalues["noise_corrected_drift"]),
        "cluster_beta": coef,
        "cluster_se": se,
        "cluster_t": t_stat,
        "cluster_t_p": cluster_t_p,
        "cluster_normal_p": float(cluster.pvalues["noise_corrected_drift"]),
        "clusters": clusters,
        "centered_spearman_rho": centered_s.statistic,
        "centered_spearman_p": centered_s.pvalue,
        "per_perturbation": per_pert,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "NA"
    if abs(value) < 0.001 and value != 0:
        return "<0.001"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            cells.append(fmt(value, digits) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, corr: pd.DataFrame, desc_task_pert: pd.DataFrame, robust: dict[str, Any]) -> None:
    overall = corr[corr["group_family"] == "overall"].iloc[0]
    by_task = corr[corr["group_family"] == "task"].sort_values("task_type")
    by_pert = corr[corr["group_family"] == "perturbation"].sort_values("perturbation_type")
    by_task_pert = corr[corr["group_family"] == "task_perturbation"].sort_values(["task_type", "perturbation_type"])
    paraphrase = corr[(corr["group_family"] == "perturbation") & (corr["perturbation_type"] == "paraphrasing")].iloc[0]
    para_cells = by_task_pert[by_task_pert["perturbation_type"] == "paraphrasing"]
    perf = desc_task_pert.sort_values(["perturbation_type", "task_type"])
    mean_para = perf[perf["perturbation_type"] == "paraphrasing"][["task_type", "mean_noise_corrected_drift", "mean_absolute_performance_change"]]

    lines = [
        "# RQ2 150-Item Expanded Analysis\n\n",
        "This report reruns the RQ2 drift-performance analysis after combining the original n50 GPT/main data with the add100 GPT/main data. The design contains 150 items per task for factual QA, math reasoning, and code generation; each item has five original generations and five generations under each of five perturbation types.\n\n",
        f"Final item-perturbation table: n = {len(df)} observations ({df['task_type'].nunique()} tasks x 150 items x 5 perturbation types). Semantic drift is measured as noise-corrected Sentence-BERT cosine distance. Performance change is original performance minus perturbed performance, so positive values indicate a performance decrease.\n\n",
        "## Overall Drift-Performance Association\n\n",
        f"Pooling all {len(df)} observations, Spearman rho = {fmt(overall['spearman_rho'])} (p = {fmt(overall['spearman_p'])}; Holm-adjusted p = {fmt(overall['spearman_holm_p'])}). Pearson r = {fmt(overall['pearson_r'])} (p = {fmt(overall['pearson_p'])}). As in the smaller RQ2 analysis, Pearson and Spearman diverge, indicating that the association is sensitive to distribution shape and high-leverage cells rather than a stable monotone relationship across all perturbations and tasks.\n\n",
        "## By-Task Association\n\n",
        markdown_table(by_task[["task_type", "n", "spearman_rho", "spearman_p", "spearman_holm_p", "pearson_r", "pearson_p"]], ["task_type", "n", "spearman_rho", "spearman_p", "spearman_holm_p", "pearson_r", "pearson_p"]),
        "\n\n",
        "## Factual QA Robustness Check\n\n",
        f"For factual QA, the raw pooled Spearman association across all perturbations is rho = {fmt(robust['raw_spearman_rho'])} (p = {fmt(robust['raw_spearman_p'])}). This is the closest direct analogue of the earlier n=50 factual QA result, but now uses 150 factual QA items x 5 perturbations = 750 observations.\n\n",
        f"After adding perturbation fixed effects, the drift coefficient is beta = {fmt(robust['ols_beta'])} with naive OLS p = {fmt(robust['ols_p'])}. With item-clustered standard errors and a t(G-1) reference distribution over G = {robust['clusters']} item clusters, beta = {fmt(robust['cluster_beta'])}, SE = {fmt(robust['cluster_se'])}, t = {fmt(robust['cluster_t'])}, p = {fmt(robust['cluster_t_p'])}. The asymptotic normal cluster p-value is {fmt(robust['cluster_normal_p'])}. Unlike the earlier 10-cluster robustness check, the expanded sample has enough clusters for the cluster-robust approximation to be less fragile.\n\n",
        f"Within-perturbation centering gives Spearman rho = {fmt(robust['centered_spearman_rho'])} (naive p = {fmt(robust['centered_spearman_p'])}), describing the residual continuous association after subtracting perturbation-type means.\n\n",
        "Per-perturbation factual QA correlations:\n\n",
        markdown_table(pd.DataFrame(robust["per_perturbation"]), ["perturbation_type", "n", "spearman_rho", "spearman_p"]),
        "\n\n",
        "## Paraphrasing Subset\n\n",
        f"Across all tasks, paraphrasing alone shows Spearman rho = {fmt(paraphrase['spearman_rho'])} (p = {fmt(paraphrase['spearman_p'])}; Holm-adjusted p = {fmt(paraphrase['spearman_holm_p'])}, n = {int(paraphrase['n'])}). Per-task paraphrasing cells are:\n\n",
        markdown_table(para_cells[["task_type", "n", "spearman_rho", "spearman_p", "spearman_holm_p", "pearson_r", "pearson_p"]], ["task_type", "n", "spearman_rho", "spearman_p", "spearman_holm_p", "pearson_r", "pearson_p"]),
        "\n\n",
        "Mean paraphrasing drift and performance change:\n\n",
        markdown_table(mean_para, ["task_type", "mean_noise_corrected_drift", "mean_absolute_performance_change"]),
        "\n\n",
        "## Aggregate Performance Change by Perturbation Type x Task\n\n",
        markdown_table(perf[["perturbation_type", "task_type", "n", "mean_noise_corrected_drift", "mean_absolute_performance_change", "drop_share"]], ["perturbation_type", "task_type", "n", "mean_noise_corrected_drift", "mean_absolute_performance_change", "drop_share"]),
        "\n\n",
        "## Summary Correlations\n\n",
        markdown_table(corr[["group_family", "task_type", "perturbation_type", "n", "spearman_rho", "spearman_p", "spearman_holm_p", "pearson_r", "pearson_p"]], ["group_family", "task_type", "perturbation_type", "n", "spearman_rho", "spearman_p", "spearman_holm_p", "pearson_r", "pearson_p"]),
        "\n\n",
        "## Answer to RQ2\n\n",
        "The expanded 150-item-per-task analysis gives a more stable estimate than the earlier 10-item-per-task RQ2 run. The pooled all-task association remains weak under Spearman, so RQ2 should not be answered as a simple global monotone relationship between semantic drift and performance decrease. The strongest evidence remains task-conditional: factual QA shows a positive drift-performance association, and the larger item count makes the perturbation-controlled, item-clustered robustness check more credible than in the original small-sample analysis. Math reasoning and code generation should be interpreted from their task-specific rows rather than inferred from the pooled trend.\n\n",
        "Paraphrasing remains the main perturbation to inspect because it tends to create larger semantic drift and larger mean performance changes than the other perturbation types, but the report keeps the multiplicity-adjusted p-values visible. The cleanest conclusion is therefore conditional rather than universal: semantic drift is most clearly informative for factual QA, while cross-task pooling masks heterogeneous behavior across tasks and perturbation families.\n\n",
        "## Output Files\n\n",
        "- `rq2_150_outputs/rq2_150_original_correctness_by_generation.csv`\n",
        "- `rq2_150_outputs/rq2_150_perturbed_correctness_by_generation.csv`\n",
        "- `rq2_150_outputs/rq2_150_performance_change_by_item.csv`\n",
        "- `rq2_150_outputs/rq2_150_drift_performance_by_item.csv`\n",
        "- `rq2_150_outputs/rq2_150_correlations.csv`\n",
        "- `rq2_150_outputs/rq2_150_descriptive_by_task_perturbation.csv`\n",
    ]
    REPORT.write_text("".join(lines), encoding="utf-8")


def validate_keys(original_rows: list[dict[str, str]], perturbed_rows: list[dict[str, str]], prompt_by_item: dict[str, dict[str, str]]) -> None:
    missing_prompts = sorted({row["item_id"] for row in original_rows + perturbed_rows} - set(prompt_by_item))
    if missing_prompts:
        raise ValueError(f"Missing prompt metadata for {len(missing_prompts)} items: {missing_prompts[:5]}")
    original_keys = [(row["item_id"], row["sample_id"]) for row in original_rows]
    perturbed_keys = [(row["item_id"], row["perturbation_type"], row["sample_id"]) for row in perturbed_rows]
    if len(original_keys) != len(set(original_keys)):
        raise ValueError("Duplicate original generation keys")
    if len(perturbed_keys) != len(set(perturbed_keys)):
        raise ValueError("Duplicate perturbed generation keys")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    original_rows, perturbed_rows, prompt_by_item = load_inputs()
    validate_keys(original_rows, perturbed_rows, prompt_by_item)

    print(f"Loaded {len(original_rows)} original generations and {len(perturbed_rows)} perturbed generations")
    eval_fields = [
        "item_id",
        "task_type",
        "dataset_name",
        "source_index",
        "source_id",
        "sample_id",
        "model_name",
        "prompt_variant",
        "perturbation_type",
        "reference_answer",
        "output_text",
        "extracted_answer",
        "performance_score",
        "factual_containment_match",
        "factual_token_f1",
        "is_correct",
        "needs_manual_review",
        "correctness_method",
        "notes",
    ]
    perf_fields = [
        "item_id",
        "task_type",
        "perturbation_type",
        "n_original_outputs",
        "n_perturbed_outputs",
        "original_performance",
        "perturbed_performance",
        "absolute_performance_change",
        "pdr",
        "performance_dropped",
    ]
    original_eval_path = OUT_DIR / "rq2_150_original_correctness_by_generation.csv"
    perturbed_eval_path = OUT_DIR / "rq2_150_perturbed_correctness_by_generation.csv"
    performance_path = OUT_DIR / "rq2_150_performance_change_by_item.csv"

    if original_eval_path.exists() and perturbed_eval_path.exists():
        original_eval = read_csv_rows(original_eval_path)
        perturbed_eval = read_csv_rows(perturbed_eval_path)
        if len(original_eval) == len(original_rows) and len(perturbed_eval) == len(perturbed_rows):
            print("Reusing cached correctness CSV files")
        else:
            code_tests = load_humanevalpack_tests()
            original_eval = evaluate_generation_rows(original_rows, prompt_by_item, code_tests, "original")
            perturbed_eval = evaluate_generation_rows(perturbed_rows, prompt_by_item, code_tests, "perturbed")
            write_csv(original_eval_path, original_eval, eval_fields)
            write_csv(perturbed_eval_path, perturbed_eval, eval_fields)
    else:
        code_tests = load_humanevalpack_tests()
        original_eval = evaluate_generation_rows(original_rows, prompt_by_item, code_tests, "original")
        perturbed_eval = evaluate_generation_rows(perturbed_rows, prompt_by_item, code_tests, "perturbed")
        write_csv(original_eval_path, original_eval, eval_fields)
        write_csv(perturbed_eval_path, perturbed_eval, eval_fields)

    if performance_path.exists():
        performance = read_csv_rows(performance_path)
        if len(performance) == 2250:
            print("Reusing cached performance-change CSV file")
        else:
            performance = build_performance_change(original_eval, perturbed_eval)
            write_csv(performance_path, performance, perf_fields)
    else:
        performance = build_performance_change(original_eval, perturbed_eval)
        write_csv(performance_path, performance, perf_fields)

    drift_rows = compute_drift_table(original_rows, perturbed_rows, performance)
    drift_fields = perf_fields + ["baseline_similarity", "perturbation_similarity", "noise_corrected_drift", "similarity_metric"]
    write_csv(OUT_DIR / "rq2_150_drift_performance_by_item.csv", drift_rows, drift_fields)

    df = pd.DataFrame(drift_rows)
    numeric = [
        "original_performance",
        "perturbed_performance",
        "absolute_performance_change",
        "pdr",
        "baseline_similarity",
        "perturbation_similarity",
        "noise_corrected_drift",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    corr = correlation_table(df)
    desc_task_pert = descriptive_table(df, ["task_type", "perturbation_type"])
    corr.to_csv(OUT_DIR / "rq2_150_correlations.csv", index=False)
    desc_task_pert.to_csv(OUT_DIR / "rq2_150_descriptive_by_task_perturbation.csv", index=False)
    descriptive_table(df, ["task_type"]).to_csv(OUT_DIR / "rq2_150_descriptive_by_task.csv", index=False)
    descriptive_table(df, ["perturbation_type"]).to_csv(OUT_DIR / "rq2_150_descriptive_by_perturbation.csv", index=False)

    robust = factual_robustness(df)
    pd.DataFrame(robust["per_perturbation"]).to_csv(OUT_DIR / "rq2_150_factual_per_perturbation_correlations.csv", index=False)
    pd.DataFrame([{k: v for k, v in robust.items() if k != "per_perturbation"}]).to_csv(OUT_DIR / "rq2_150_factual_robustness.csv", index=False)
    write_report(df, corr, desc_task_pert, robust)
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
