#!/usr/bin/env bash
# Claude Sonnet 5 frontier arm: both arms of full_3x via the Anthropic Message
# Batches API, thinking disabled (mirrors "no reasoning"), 2k output tokens.
# Sonnet 5 rejects temperature/top_p, so no sampling params are sent.
# Run: bash pipeline/run_sonnet5_batch.sh
# Resumable: re-run after an interruption and it skips completed prompt_ids.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Load keys from repo-root .env if present, else fall back to ~/.anthropic_key.
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
: "${ANTHROPIC_API_KEY:=$(cat ~/.anthropic_key 2>/dev/null || true)}"
export ANTHROPIC_API_KEY
cd "$(dirname "$0")"

# Frontier arm: 1 rep (rep1 files) at an 8k cap (effectively uncapped -- 0%
# truncation vs the OS models' 0.3-3.6%). 1-rep is justified by IssueBench
# Appendix H (sampling once per prompt has negligible impact on issue-level stance).
MODEL=claude-sonnet-5
COMMON="--model $MODEL --thinking disabled --max-tokens 8000 --poll-seconds 60"
OUT=../results/full_3x

echo "=== ${MODEL} Arm A : $(date) ==="
python3 run_anthropic_batch.py --prompts ../data/processed/full_3x/prompts_arm_a_rep1.csv $COMMON \
  --out-jsonl "${OUT}/gen_sonnet5_arm_a.jsonl" --out-csv "${OUT}/gen_sonnet5_arm_a.csv"

echo "=== ${MODEL} Arm B : $(date) ==="
python3 run_anthropic_batch.py --prompts ../data/processed/full_3x/prompts_arm_b_rep1.csv $COMMON \
  --out-jsonl "${OUT}/gen_sonnet5_arm_b.jsonl" --out-csv "${OUT}/gen_sonnet5_arm_b.csv"

echo "=== ${MODEL} DONE : $(date) ==="
