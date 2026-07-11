#!/usr/bin/env bash
# Data-parallel vLLM generation: one shard (both arms) on one GPU. Run once per
# shard on a separate GPU, then merge the per-shard files. Usage:
#   bash pipeline/run_gen_vllm_shard.sh <model_path> <out_prefix> <num_shards> <shard_index> [extra args...]
set -euo pipefail

MODEL="$1"; PREFIX="$2"; NSHARDS="$3"; SIDX="$4"; shift 4
VENV=/data/kell8360/vllm017-venv/bin/python
IN=data/processed/full_3x
OUT=/data/kell8360/full_3x_out
mkdir -p "$OUT"

for arm in a b; do
  echo "=== ${PREFIX} arm_${arm} shard ${SIDX}/${NSHARDS} : $(date) ==="
  "$VENV" pipeline/vllm_gen.py \
    --prompts "${IN}/prompts_arm_${arm}.csv" \
    --model "$MODEL" \
    --max-new-tokens 2000 \
    --num-shards "$NSHARDS" --shard-index "$SIDX" \
    --out-jsonl "${OUT}/${PREFIX}_arm_${arm}.s${SIDX}.jsonl" \
    --out-csv   "${OUT}/${PREFIX}_arm_${arm}.s${SIDX}.csv" \
    "$@"
done
echo "=== ${PREFIX} shard ${SIDX} DONE : $(date) ==="
