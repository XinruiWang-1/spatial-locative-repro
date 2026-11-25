# Spatial Locative Reference Resolution on ALFRED

**English** | [中文版](README.zh.md)

Reproducibility package for the course project "How much does a spatial description
actually constrain? Two rulers for grounded instruction following".

The task: given an instruction such as *"Pick up the pen that is on the left side of the
dresser"* in a scene with 2 to 7 interchangeable pens, decide which pen the instruction
refers to. The repository holds the frozen sample table, the geometry reference
implementation, all baselines, the 7B baseline, and the scripts that regenerate every
table and figure.

Two commands reproduce everything (CPU only, about 20 s):

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

---

## 1. Research question

Spatial locative expressions ("the cup to the left of the sink") are usually reported as
one accuracy number. That number mixes two different questions:

1. Does the description constrain the scene at all? That is, is the human annotation
   inside the region the words describe?
2. Can a method pick exactly one object? A robot has to grasp one cup.

We measure (1) with a set-valued ruler (U-SET) and (2) with a single-point selector
(`M-soft`). The gap between the two comes from the data, not from a failure of the
method.

## 2. Reported results

665 scorable samples, 2.8 candidates on average. All comparisons are paired (McNemar
exact test, 4,000 bootstrap resamples for confidence intervals).

| Subset | n | U-SET (set ruler) | M-soft (ours) | M2 hard filter | M0 nearest | 7B (best prompt) | random in pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 all scorable | 665 | **88.4%** | **68.4%** | 62.7% | 64.8% | 57.4% | 38.3% |
| S2 direction computable | 596 | 87.1% | **72.5%** | 66.1% | 68.5% | 61.1% | — |
| S3 only truly filtered | 515 | 85.0% | **72.8%** | 65.4% | 68.3% | 61.6% | — |

Three findings.

1. Direction should stay a soft prior. Hard-filtering the wrong side drops accuracy from
   68.4% to 62.7% (p < 0.0001), because 157 of the 665 human annotations sit on the side
   opposite to the words.
2. The distance rule is almost never tested. Only 8 of 665 instructions (1.2%) mention
   nearest or closest, and in 4 of those the reference object has usable coordinates, so
   the rule fires on 4 samples (0.6%). Distance is a geometric prior we chose, not a
   constraint carried by the sentence.
3. A 7B LLM keeps the direction but cannot rank inside it. Qwen2.5-7B-Instruct picks an
   object in the allowed half-plane 98.6% of the time, higher than the annotations' own
   88.4%, yet reaches only 57.4% overall and loses to a one-line "nearest" rule. Three
   prompt variants do not differ significantly (p = 0.22 and p = 0.68).

![Main subsets](figures/paper_fig1_subsets.png)
![7B baseline](figures/paper_fig5_llm.png)

## 3. Repository structure

```
spatial-locative-repro/
├── README.md                  this file
├── README.zh.md               Chinese translation of this file
├── requirements.txt           core dependencies (CPU)
├── requirements-llm.txt       only needed to re-run 7B inference (GPU)
├── LICENSE                    MIT (our code) + attribution of third-party assets
├── AI_DISCLOSURE.md           what an AI assistant helped with
├── AI_DISCLOSURE.zh.md        Chinese translation
├── docs/                      the same README and disclosure as Word files
├── configs/
│   ├── main.yaml              weights, subsets, bootstrap settings
│   └── llm_7b.yaml            model, decoding, prompts, hardware
├── data/
│   ├── README.md              provenance, band construction, integrity check
│   ├── clean_spatial_ext.csv  frozen samples (1,392 rows, 665 scorable)
│   ├── DATA_LOCK.json         sha256 of every frozen asset
│   ├── llm_prompts.jsonl      665 samples x 3 prompt variants
│   └── derived/               loc_candidates.json.gz, loc_anchors.json.gz
├── src/
│   ├── _paths.py              all path resolution (no absolute paths anywhere)
│   ├── scoring.py             geometry and cascade scoring -> results/cascade_detail.csv
│   ├── evaluate.py            tables T1-T12 and 5 figures -> results/, figures/
│   ├── selection.py           the proposed selector (soft prior w=0.2 + distance layer)
│   ├── baselines.py           control suite, weight sweep, cross-validation
│   ├── text_baseline.py       TF-IDF text baseline and its tokeniser/synonyms
│   ├── build_dataset.py       data prep from the raw ALFRED release (optional)
│   ├── build_anchors.py       anchor extraction (optional)
│   ├── extend_dataset.py      adds bands D0/D1/D2/D2R (optional)
│   ├── audit_data.py          final consistency audit (optional)
│   ├── inference_llm.py       7B inference, HF backend (GPU; predictions included)
│   └── synonyms.py            33-entry synonym table (single source of truth)
├── scripts/
│   ├── run_all.sh             the main experiment, end to end
│   └── run_llm_7b.sh          the 7B baseline
├── results/                   every table in the paper (regenerated by run_all.sh)
└── figures/                   the five figures in the paper
```

## 4. Hardware

| Component | Requirement |
|---|---|
| Main experiment (`scripts/run_all.sh`) | any CPU, < 1 GB RAM, about 20 s |
| 7B baseline (`scripts/run_llm_7b.sh`) | 1 x NVIDIA RTX A5000 (24 GB), about 15.2 GB peak |
| Disk | about 8 MB of files, 15 MB with the git history; about 36 GB more if you also download the raw ALFRED release |

The reported 7B numbers were produced on the course-provided RTX A5000 (24 GB), in
bfloat16, with greedy decoding and batch size 8.

## 5. Software and configuration

Tested with Python 3.15.0 on Windows 11. The code does plain pandas, numpy and
matplotlib arithmetic and contains no OS-specific paths, but it has not been run on
Linux or macOS.

```
numpy==2.5.4
pandas==3.0.6
scikit-learn==1.9.1
matplotlib==3.11.2
```

The LLM stack in `requirements-llm.txt` is only needed to regenerate the predictions:
`torch==2.14.1`, `transformers==5.24.0`, `accelerate==1.11.0`, `bitsandbytes==0.48.0`
(CUDA 13.0 in the reported run).

### How the configuration files are used

`configs/*.yaml` record the constants; they are not a runtime switch. No script accepts a
`--config` argument: `src/scoring.py` and `src/evaluate.py` take no options at all, and
the values written in the YAML are the literals used in `src/selection.py` and
`src/evaluate.py`. One dataset, one rule and one hyper-parameter do not justify a config
loader, but the settings still need to be readable in one place, which is what the two
files are for. If you want to try a different `direction_weight`, edit `W_ORI` in
`src/selection.py` line 41 and re-run `scripts/run_all.sh`.

Randomness is confined to the bootstrap. All four seeds are fixed in the source:

| Seed | Location | What it resamples |
|---|---|---|
| 0 | `src/evaluate.py` -> `ci(p, seed=0)` | confidence intervals of the main table |
| 1 | `src/baselines.py` -> `boot_ci(ok, seed=1)` | the U-SET set-ruler interval |
| 2 | `src/baselines.py` -> `boot_ci(p, seed=2)` | every single-point method inside `summarize()` |
| 3 | `src/baselines.py` -> `boot_ci(hits, seed=3)` | the weight-sweep control experiment |

`configs/main.yaml` names seed 0 as `bootstrap_seed` because that is the one behind the
intervals reported for the main table. Nothing else in the pipeline draws a random
number: the table is frozen, the selector is a deterministic argmin, and the 7B run is
greedy (`do_sample: false`). `run_all.sh` therefore writes the same CSV on any machine,
and there is no seed to tune.

## 6. Installation

```bash
git clone https://github.com/XinruiWang-1/spatial-locative-repro.git
cd spatial-locative-repro
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Dataset

All experiments use a derived subset of ALFRED (Shridhar et al., CVPR 2020, MIT License,
https://github.com/askforalfred/alfred). The frozen sample table ships with this
repository, so reproducing the paper needs no download. The 36 GB raw release is only
needed if you want to rebuild the table from scratch.

Band construction, geometry definitions, integrity hashes and regeneration commands are
documented in [`data/README.md`](data/README.md). Summary:

* 1,392 rows in total; `Scorable == 1` leaves **665** samples.
* 523 exact object anchors, 73 scene-level anchors, 69 with no anchor (text fallback).
* Excluded: 225 single-candidate pools (nothing to choose) and 502 samples whose anchor
  is absent from the scene graph.
* Three precomputed columns drive everything: `Side_Cos` (which half-plane the candidate
  is on), `D_Target` (distance **to the anchor**), `Label_In_Side` (whether the human
  annotation itself obeys the words).

## 8. Method in five lines

```python
# for each candidate of the target category
side = cos(robot -> anchor, anchor -> candidate)       # > 0  => named side
dist = ||candidate - anchor||                          # metres, x-z plane
score = dist * (1 - 0.2 * sign_positive(side))         # soft prior, w = 0.2
pick  = argmin(score)                                  # ties -> higher side score
# the distance layer becomes a hard filter only when the instruction says "nearest"
```

The implementation is `src/selection.py`; the constants are recorded in
`configs/main.yaml`.

## 9. Training

There is no training step. The method is a deterministic scoring function with one
hyper-parameter, `direction_weight = 0.2`, chosen by a 5-fold out-of-fold sweep in
`src/baselines.py`. The same value is best in every fold. The 7B model is used zero-shot,
with no fine-tuning and no adapter.

The repository does not ship a `train.py`. A placeholder would suggest a trained
component that does not exist. The contribution here is a measurement protocol and a
rule; a trained component would be a different claim.

## 10. Evaluation

```bash
bash scripts/run_all.sh
# or, step by step
python src/scoring.py      # cascade scoring -> results/cascade_detail.csv
python src/evaluate.py     # all tables and 5 figures
python src/evaluate.py --no-fig    # skip plotting
```

`src/evaluate.py` writes one CSV per table and also `results/PAPER_EXPERIMENTS.md`, the
full table-by-table report in Chinese.

| File | Content |
|---|---|
| `paper_T1_main.csv` | the main table: 3 subsets x 19 methods |
| `paper_T2/T3/T4` | difficulty bands, relation types, allowed-set size |
| `paper_T5_audit.csv` | annotation audit: 439 self-consistent vs 157 contradictory |
| `paper_T6_ruler.csv`, `paper_T6b_quadrant.csv` | the two rulers and the 2x2 bridge between them |
| `paper_T7_ablation.csv` | soft prior vs hard filter vs no direction layer |
| `paper_T8/T8b` | evolution of the pipeline; two-stage (category x instance) decomposition |
| `paper_T9_failures.csv` | where the remaining errors come from |
| `paper_T10/T11` | weight sweep; paired significance tests |
| `paper_T12a-d` | 7B baseline: tests, behaviour diagnosis, strata, position bias |
| `paper_detail.csv` | per-sample trace for all 665 samples |

## 11. Expected results

After `bash scripts/run_all.sh`, `results/paper_T1_main.csv` must contain:

| Method | S1 (665) | S2 (596) | S3 (515) |
|---|---:|---:|---:|
| U-SET (set ruler) | 88.4 | 87.1 | 85.0 |
| M-soft (ours) | 68.4 | 72.5 | 72.8 |
| M2 hard filter | 62.7 | 66.1 | 65.4 |
| M0 nearest | 64.8 | 68.5 | 68.3 |
| LLM 7B · v2 | 57.4 | 61.1 | 61.6 |

If any value differs, the frozen data was modified. Check it against the sha256 snippet
in `data/README.md`.

## 12. The 7B baseline

The 7B numbers are final and already in the tables: row `LLM 7B · v2` of
`paper_T1_main.csv`, plus the four diagnostics in `paper_T12a-d`. That is
Qwen2.5-7B-Instruct with 1,995 greedy generations (665 samples times 3 prompt variants)
on one A5000. Nothing in this section is needed to reproduce the reported results; it
only regenerates the predictions.

```bash
pip install -r requirements-llm.txt
bash scripts/run_llm_7b.sh          # writes results/llm_predictions.jsonl
bash scripts/run_all.sh             # merges the predictions into T1 and T12
```

* Model: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0), 15.2 GB in bf16.
* The predictions from the reported run are committed, so the tables reproduce without a
  GPU. Only regenerating them needs the A5000.
* Three prompt variants: raw numbers, verbalised relations, verbalised plus an explicit
  two-step rule.

## 13. Limitations

* One dataset (ALFRED) and one LLM (Qwen2.5-7B, single greedy run, no CoT, no
  self-consistency). The LLM result is a negative baseline for this model, not a claim
  about LLMs in general.
* 68.4% is a given-category number. It does not answer "is the object a pen". End to end,
  where the category is not given and a TF-IDF text score competes with distance over the
  full pool, is 38.9%.
* There is no closed-loop rollout in AI2-THOR. The metric is "chose the right object",
  not task success.
* Method names in the CSV files are Chinese, the author's working language. The mapping
  is in `results/README.md`.

## 14. Citation and attribution

If you use this code or the derived tables:

```bibtex
@misc{wang2026spatiallocative,
  author = {Wang, Xinrui},
  title  = {How much does a spatial description constrain? Two rulers for grounded
            instruction following},
  year   = {2026},
  note   = {Course project, reproducibility package}
}
```

Third-party assets, all attributed in `LICENSE`:

* ALFRED: Shridhar et al., CVPR 2020, MIT License.
* Qwen2.5-7B-Instruct: Apache-2.0.
* scikit-learn, pandas, numpy, matplotlib: used as libraries, no code copied.

## 15. AI use

An AI coding assistant was used for refactoring, documentation and debugging. It did not
design the method and did not produce any number. See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md)
(Chinese: [`AI_DISCLOSURE.zh.md`](AI_DISCLOSURE.zh.md)).

## 16. License

MIT for the code and the derived tables. The ALFRED raw release and images are not
redistributed here.

## 17. Reproduction checklist

The rubric asks whether a reader can follow one chain: repository, environment, data,
configuration, inference, evaluation, reported result. Each link and where to find it:

| Question | Answer |
|---|---|
| Where is the complete source code? | `src/`, 12 modules with one job each, listed in §3 |
| How is the environment installed? | §6: `pip install -r requirements.txt`, four pinned versions |
| Where does the dataset come from? | §7 and `data/README.md`: ALFRED, CVPR 2020, MIT, official source; nothing to download |
| How is the data prepared? | `data/README.md`: how the 665 scorable samples were cut; rebuild scripts optional |
| Which configuration reproduces the main experiment? | §5: `configs/main.yaml` records `W_ORI = 0.2`; no script takes a flag |
| How is the model trained? | §9: no training step; the one hyper-parameter comes from a 5-fold sweep |
| How is the model evaluated? | §10: `bash scripts/run_all.sh`, about 20 s on CPU |
| Where are the metrics stored? | `results/paper_T*.csv`, one file per table, plus `paper_detail.csv` per sample |
| Do they match the paper? | §11 gives the five values to compare in `paper_T1_main.csv` |
| Is the A5000 required? | No. Only regenerating the 7B predictions needs it; they are committed |

The whole chain, on a machine with no GPU:

```bash
git clone https://github.com/XinruiWang-1/spatial-locative-repro.git
cd spatial-locative-repro
pip install -r requirements.txt
bash scripts/run_all.sh
cat results/paper_T1_main.csv
```

If step 4 ends without an error and step 5 shows `U-SET` at 88.4 and `M-soft` at 68.4 on
S1, the submission reproduces end to end.
