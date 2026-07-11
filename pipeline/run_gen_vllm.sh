#!/usr/bin/env bash
# vLLM generation for one model, both arms, 2k tokens, 3 reps. Outputs to /data
# (not /home, which is near-full). Usage:
#   bash pipeline/run_gen_vllm.sh <model_path> <out_prefix> [extra vllm_gen args...]
set -euo pipefail

MODEL="$1"; PREFIX="$2"; shift 2
VENV=/data/kell8360/vllm017-venv/bin/python
IN=data/processed/full_3x
OUT=/data/kell8360/full_3x_out
mkdir -p "$OUT"

for arm in a b; do
  echo "=== ${PREFIX} arm_${arm} : $(date) ==="
  "$VENV" pipeline/vllm_gen.py \
    --prompts "${IN}/prompts_arm_${arm}.csv" \
    --model "$MODEL" \
    --max-new-tokens 2000 \
    --out-jsonl "${OUT}/${PREFIX}_arm_${arm}.jsonl" \
    --out-csv   "${OUT}/${PREFIX}_arm_${arm}.csv" \
    "$@"
done
echo "=== ${PREFIX} DONE : $(date) ==="
