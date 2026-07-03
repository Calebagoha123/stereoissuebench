#!/usr/bin/env bash
# Run both arms of the fresh 3-repeat generation for one model at 2k max tokens.
# Usage: bash pipeline/run_gen_model.sh <model_path> <out_prefix>
#   e.g. bash pipeline/run_gen_model.sh /data/.../models--Qwen--Qwen3.6-27B gen_qwen
set -euo pipefail

MODEL="$1"
PREFIX="$2"
D=data/processed/full_3x
COMMON=(--device cuda:0 --batch-size 8 --max-new-tokens 2000)

for arm in a b; do
  echo "=== ${PREFIX} arm_${arm} : $(date) ==="
  python pipeline/02_run_generation.py \
    --prompts "${D}/prompts_arm_${arm}.csv" \
    --model "${MODEL}" \
    "${COMMON[@]}" \
    --out-jsonl "${D}/${PREFIX}_arm_${arm}.jsonl" \
    --out-csv   "${D}/${PREFIX}_arm_${arm}.csv"
done
echo "=== ${PREFIX} DONE : $(date) ==="
