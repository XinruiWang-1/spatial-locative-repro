# AI-Use Disclosure

This project was developed with the help of an AI coding assistant (WorkBuddy,
powered by a large language model). In the interest of research integrity, here is
exactly what that help consisted of.

## What the AI assistant was used for

| Area | Extent |
|---|---|
| Code refactoring for this repository | Moving scripts into `src/`, replacing hard-coded paths with `src/_paths.py`, renaming modules, writing `scripts/*.sh` |
| Documentation | Drafting this README, `data/README.md`, `AI_DISCLOSURE.md`, the YAML configs, and code comments added during the move |
| Debugging | Locating defects in the evaluation code (e.g. a `Series.mode` misuse that leaked object reprs into a results table) |
| Language | Translating Chinese working notes into the English text of this README |

## What the AI assistant did **not** do

* It did not design the method, choose the baselines, or decide which experiments to run.
* It did not produce, edit, or "fix" any reported number. Every number in `results/`
  is written by `src/scoring.py` and `src/evaluate.py` from the frozen data, and can be
  regenerated with `bash scripts/run_all.sh`.
* It did not annotate data. The labels are the original ALFRED annotations.

## Verification performed by the author

* The refactored repository reproduces the pre-refactor tables **byte for byte**
  (`results/paper_T1_main.csv`, `paper_T7_ablation.csv`, `paper_T11_tests.csv`,
  `paper_detail.csv` were `diff`-ed against the originals).
* `data/clean_spatial_ext.csv` is unchanged: sha256 is checked against
  `data/DATA_LOCK.json`.
* The author can explain every submitted script and re-run every experiment.

## Third-party code and models

No third-party source code was copied into `src/`. The following are used as libraries
or as data/models and are attributed in `README.md` and `LICENSE`:

* scikit-learn (`TfidfVectorizer`, cosine similarity) — BSD-3-Clause
* pandas, numpy, matplotlib — BSD-3-Clause / PSF
* ALFRED dataset — MIT (https://github.com/askforalfred/alfred)
* Qwen2.5-7B-Instruct — Apache-2.0 (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
