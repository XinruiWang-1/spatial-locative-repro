# Results

Every file here is written by `src/scoring.py` or `src/evaluate.py`. Re-run
`bash scripts/run_all.sh` to regenerate all of them from `data/clean_spatial_ext.csv`.
The one exception is `llm_predictions.jsonl`, which is the output of the 7B model and is
committed so that the LLM row reproduces without a GPU.

## Table → paper mapping

| File | Paper element |
|---|---|
| `paper_T1_main.csv` | main table: 3 subsets × 17 methods |
| `paper_T2_band.csv` | accuracy by difficulty band (pool size / allowed-set size) |
| `paper_T3_rel.csv` | accuracy by relation type (*left/right*, *front/behind*, …) |
| `paper_T4_pool.csv` | accuracy by allowed-set size 1–4 (kills the "noise ceiling" claim) |
| `paper_T5_audit.csv` | annotation audit: self-consistent vs contradictory subsets |
| `paper_T6_ruler.csv` | the two rulers on the same samples |
| `paper_T6b_quadrant.csv` | 2×2 bridge: in/out of the allowed region × correct/wrong pick |
| `paper_T7_ablation.csv` | soft prior vs hard filter vs no direction layer |
| `paper_T8_evolution.csv` | pipeline evolution (text → formula → formula+direction → ours) |
| `paper_T8b_twostage.csv` | two-stage decomposition (category recognition × instance choice) |
| `paper_T9_failures.csv` | attribution of the remaining errors |
| `paper_T10_sweep.csv` | direction-weight sweep |
| `paper_T11_tests.csv` | paired McNemar tests against every baseline |
| `paper_T12a_llm_tests.csv` | 7B: paired tests vs M\*\*, M2, M0 and across prompts |
| `paper_T12b_llm_behaviour.csv` | 7B: in-allowed-set rate, agreement with "nearest", position bias |
| `paper_T12c_llm_strata.csv` | 7B: accuracy by stratum |
| `paper_T12d_llm_position.csv` | 7B: answer distribution over option slots A–D |
| `paper_detail.csv` | per-sample trace for all 665 samples (one row each) |
| `cascade_detail.csv` | intermediate: which stage decided each sample |
| `cascade_main.csv`, `CASCADE_REPORT.md` | stage-1 scoring summary (Chinese report) |
| `PAPER_EXPERIMENTS.md` | full table-by-table report (Chinese, author's working language) |
| `llm_predictions.jsonl` | 665 × 3 model answers, one JSON object per line |

## Column labels are Chinese — here is the mapping

The tables keep the method names used during development. If you are reading
`paper_T1_main.csv`, this is what each row is:

| Label in the CSV | English |
|---|---|
| `ORACLE 直接取标注` | upper bound — return the human annotation (not a method) |
| `U-SET 允许集合判定（口径）` | **set ruler**: is the annotation inside the region the words describe |
| `★ M** 软先验 0.2 + 距离硬层` | **proposed**: soft direction prior (w = 0.2) + distance layer |
| `M3* 软加权 w=0.2（无硬层）` | ablation: same prior, distance layer removed |
| `M3 软加权 w=0.5` | stronger direction weight (older setting) |
| `M3a 硬筛后软加权` | hard filter, then soft re-weighting |
| `M2 硬筛 + 侧内最近` | hard filter on the named side, then nearest within it |
| `M0 取锚点最近` | nearest to the anchor, direction ignored |
| `U-COS 允许集内最靠侧` | most lateral candidate inside the allowed set |
| `U-RAND 允许集内随机` | random inside the allowed set (expectation) |
| `R 取机器人最近` | nearest to the robot (sanity baseline) |
| `FAR 取机器人最远` | farthest from the robot (sanity baseline) |
| `RND 全池随机` | random over the full candidate pool (expectation) |
| `A 纯文本 TF-IDF（全池）` | text only, TF-IDF over the full pool |
| `B 公式 tfidf×距离（全池）` | TF-IDF × distance, full pool |
| `B2 公式 + 方位（全池）` | TF-IDF × distance + direction, full pool |
| `LLM 7B · v1 只给数字` | Qwen2.5-7B, raw coordinate numbers |
| `LLM 7B · v2 说成人话` | Qwen2.5-7B, relations verbalised |
| `LLM 7B · v3 明写两步规则` | Qwen2.5-7B, verbalised + explicit two-step rule |

Subsets: `S1 全量` = all 665 scorable; `S2 方向可算` = 596 with a computable direction;
`S3 真被筛过` = 515 after removing empty-filter fallbacks.
`性质` marks whether a row is a single-point answer (`单点`) or a set-valued judgement
(`集合`) — the two are not comparable head-to-head, which is the point of the paper.
