# RQ2_150 Resource Bundle

This folder is a copy of the local resources used to produce `RQ2_150.md`.

## Contents

- `markdown/`
  - Final reports: `RQ2_150.md` and `RQ2_150_zh.md`.
- `scripts/src/`
  - `65_analyze_rq2_150.py`: main script that builds correctness tables, drift-performance tables, summary CSVs, and `RQ2_150.md`.
  - `23_evaluate_rq2_correctness.py`: general RQ2 correctness scoring script.
  - `60_prepare_add100_three_task_prompts.py` to `64_apply_paraphrase_repair_outputs.py`: add100 prompt/output preparation, validation, and repair scripts used before the expanded RQ2 analysis.
  - `requirements.txt`: project dependency list copied from the repo root.
- `data/inputs/`
  - Generation CSVs and prompt CSVs read by `65_analyze_rq2_150.py`.
  - add100 validation and paraphrase repair reports/data.
- `data/outputs/rq2_150_outputs/`
  - CSV outputs generated for the final report, including correctness, performance-change, drift-performance, correlation, and descriptive tables.
- `data/external/huggingface/datasets/bigcode___humanevalpack/python/`
  - Local HumanEvalPack cache files used by code-generation functional correctness scoring.

No API key files are included.
