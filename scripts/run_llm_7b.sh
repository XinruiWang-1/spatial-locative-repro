#!/usr/bin/env bash
# Optional: re-run the 7B baseline. Needs a 24 GB GPU and ~16 GB of weights.
# The shipped results/llm_predictions.jsonl was produced with exactly this command.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}

for MODE in v1_numeric v2_verbal v3_cascade; do
  echo "=== $MODE"
  "$PY" src/inference_llm.py \
    --model "$MODEL" \
    --prompts data/llm_prompts.jsonl \
    --mode "$MODE" \
    --out "results/llm_predictions.jsonl" \
    --batch 8 \
    --max-new-tokens 16
done

echo "Now re-run scripts/run_all.sh to merge the new predictions into the tables."
