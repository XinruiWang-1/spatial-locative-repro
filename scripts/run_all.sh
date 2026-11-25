#!/usr/bin/env bash
# Reproduce every table and figure reported in the paper.
# CPU only, ~10 s on a laptop. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}

echo "[1/2] geometry + cascade scoring  -> results/cascade_*.csv"
"$PY" src/scoring.py

echo "[2/2] tables T1-T12 + 5 figures   -> results/*.csv, figures/*.png"
"$PY" src/evaluate.py

# The LLM row needs results/llm_predictions.jsonl (committed). Without it the
# evaluation still runs, but T12 silently disappears — so fail loudly instead.
if ! grep -q "LLM 7B" results/paper_T1_main.csv; then
  echo
  echo "WARNING: results/llm_predictions.jsonl is missing, so the 7B rows were skipped."
  echo "         Restore it with: git checkout results/llm_predictions.jsonl"
  echo "         or re-run the model: bash scripts/run_llm_7b.sh"
fi

echo
echo "Done. Check results/paper_T1_main.csv against the numbers in README.md."
