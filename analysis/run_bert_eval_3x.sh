#!/usr/bin/env bash
# DeBERTa stance scoring for the 3-repeat 2k-token generations (both arms, 3 OS
# models). Self-contained for a raw nohup launch on Brains (no GPU allocator):
# uses the project venv python directly and defaults to CPU so it never touches a
# contended GPU. Pass "cuda:0" as $1 to run on a pinned GPU instead.
# Reads/writes the big per-arm files in /data; slims into repo results/full_3x/.
set -euo pipefail
cd /home/kell8360/stereoissuebench

PY=.venv/bin/python
DEVICE="${1:-cpu}"
OUT=/data/kell8360/full_3x_out
MODELDIR=data/processed/stance_model/final_model

# CPU threading: use a chunk of cores but leave headroom on the shared box.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
[[ "$DEVICE" == cpu ]] && export CUDA_VISIBLE_DEVICES=""

for m in llama gemma qwen; do
  for arm in a b; do
    echo "=== bert ${m} arm_${arm} device=${DEVICE} : $(date) ==="
    "$PY" stance_model/predict.py \
      --generations "${OUT}/gen_${m}_arm_${arm}.jsonl" \
      --model-dir "${MODELDIR}" \
      --out "${OUT}/bert_${m}_arm_${arm}.csv" \
      --device "${DEVICE}" --batch-size 64
  done
done

echo "=== combining into slim bert_eval_<model>.csv : $(date) ==="
"$PY" analysis/combine_bert_eval.py --in-dir "${OUT}" --out-dir results/full_3x

echo "=== BERT EVAL ALL DONE : $(date) ==="
