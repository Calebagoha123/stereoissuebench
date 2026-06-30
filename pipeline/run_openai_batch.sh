#!/usr/bin/env bash
# Submit both arms to the OpenAI Batch API safely (sequential shards, adaptive
# sizing, auto reassemble). Run: bash pipeline/run_openai_batch.sh
# Resumable: re-run after an interruption and it skips completed prompt_ids.
set -euo pipefail
export OPENAI_API_KEY="$(cat ~/.openai_key)"
cd "$(dirname "$0")"

MODEL=gpt-5.4-mini-2026-03-17
COMMON="--model $MODEL --reasoning-effort none --max-completion-tokens 1000 --poll-seconds 60"

echo "=== Arm A ==="
python3 run_openai_batch.py --prompts ../data/processed/full/prompts_arm_a.csv $COMMON \
  --out-jsonl ../results/full/gen_openai_arm_a.jsonl --out-csv ../results/full/gen_openai_arm_a.csv

echo "=== Arm B ==="
python3 run_openai_batch.py --prompts ../data/processed/full/prompts_arm_b.csv $COMMON \
  --out-jsonl ../results/full/gen_openai_arm_b.jsonl --out-csv ../results/full/gen_openai_arm_b.csv

echo "All done."
