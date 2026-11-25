# Data

The repository ships the **derived** sample table used by every experiment. You do
**not** need to download ALFRED to reproduce the reported numbers.

## What is included

| File | Size | Rows | Role |
|---|---:|---:|---|
| `clean_spatial_ext.csv` | 400 KB | 1,392 | frozen sample table, sha256 `336aaded…425c1` |
| `DATA_LOCK.json` | 4.6 KB | — | sha256 / byte size / row count of every frozen asset |
| `derived/loc_candidates.json.gz` | 3.4 MB | — | per-task movable-object instances with (x, z) coordinates |
| `derived/loc_anchors.json.gz` | 942 KB | — | per-task anchor coordinates + robot pose |
| `llm_prompts.jsonl` | 1.9 MB | 1,995 | 665 samples × 3 prompt variants (no answers) |

`results/llm_predictions.jsonl` (665 × 3 model answers) ships with the repo, so the
LLM row of the main table also reproduces without a GPU.

## Where the data comes from

Everything is derived from the **ALFRED** dataset:

* Shridhar et al., *ALFRED: A Benchmark for Interpreting Grounded Instructions for
  Everyday Tasks*, CVPR 2020.
* Official source: https://github.com/askforalfred/alfred (MIT License)
* Raw release used: `json_2.1.0/` + `json_feat_2.1.0/` from the official download
  script in that repository (`download_data.sh`). About 36 GB — **not** included here.

## How the 665 scorable samples were selected

The frozen table has 1,392 rows; only `Scorable == 1` (665 rows) enters the experiment.

| Band | n | Anchor status | Direction computable | Scorable |
|---|---:|---|---|:---:|
| A / B / C | 523 | exact object anchor | yes | ✅ |
| D2R | 73 | scene-level anchor | yes | ✅ |
| D1 | 69 | no anchor mentioned | no | ✅ (falls back to text) |
| D0 | 225 | single candidate in the pool | — | ❌ nothing to choose |
| D2 | 502 | anchor not in the scene graph | no | ❌ |

So the 665 = 523 exact + 73 scene + 69 unanchored, and 596 of them have a computable
direction (S2); 515 of those survive the empty-filter fallback check (S3).

Three geometric quantities are precomputed per row and reused everywhere:

* `Side_Cos` — cosine between *robot → anchor* and *anchor → candidate*; `> 0` means the
  candidate lies on the half-plane named by the instruction.
* `D_Target` — Euclidean distance (x-z plane) from the candidate to the **anchor**
  (not to the robot; this distinction matters, see the paper's method section).
* `Label_In_Side` — whether the human annotation itself falls on the named half-plane.

## Regenerating from scratch (optional)

Only needed if you want to rebuild the table from the raw release. Expect a few hours
and ~36 GB of disk:

```bash
# 1. get the raw release (official script, ~36 GB)
git clone https://github.com/askforalfred/alfred.git
cd alfred && bash download_data.sh && cd ..

# 2. rebuild the sample table and the geometry assets (run from the repo root)
python src/build_dataset.py --root /path/to/json_2.1.0 \
       --out-dataset data/dataset_locative.csv \
       --out-cand    data/loc_candidates.json \
       --out-anchor  data/loc_anchors.json
python src/build_anchors.py        # scene-level anchors -> data/scene_anchors.json
python src/extend_dataset.py       # adds bands D0/D1/D2/D2R -> data/clean_spatial_ext.csv
python src/audit_data.py           # final consistency audit -> results/FINAL_CLEAN_AUDIT.md
```

`src/build_dataset.py` is the only script that needs the raw release; the others read
the intermediate files in `data/`.

These scripts are documented for provenance and are **not** part of the reproduction
path: the frozen CSV in this repo is the artifact every experiment reads.

## Integrity check

```bash
python - <<'PY'
import hashlib, json, pathlib
lock = json.load(open("data/DATA_LOCK.json", encoding="utf-8"))
for f in lock["files"]:
    p = pathlib.Path("data") / f["path"]
    if not p.exists():
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(("OK  " if h == f["sha256"] else "FAIL"), f["path"])
PY
```

## Terms

The derived tables are redistributed under the repository's MIT license. They are
derived from ALFRED annotations and remain subject to ALFRED's MIT terms. Scene images
and the 36 GB raw release are **not** redistributed here — download them from the
official source above.
