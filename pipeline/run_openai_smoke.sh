#!/usr/bin/env bash
# 20-row smoke test for the OpenAI generation arm. Run: bash pipeline/run_openai_smoke.sh
set -euo pipefail
export OPENAI_API_KEY="$(cat ~/.openai_key)"
cd "$(dirname "$0")"
python3 02_run_generation_openai.py \
  --prompts ../data/processed/full/prompts_arm_a.csv \
  --model gpt-5.4-mini-2026-03-17 \
  --reasoning-effort none \
  --limit 20 --concurrency 5 --overwrite \
  --out-jsonl /tmp/smoke.jsonl --out-csv /tmp/smoke.csv
