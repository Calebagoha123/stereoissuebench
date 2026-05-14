#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${THESIS_PIPELINE_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${THESIS_PIPELINE_DATA_ROOT}"
elif [[ -d "/data/kell8360/thesis_framing_pipeline" ]]; then
  DATA_ROOT="/data/kell8360/thesis_framing_pipeline"
else
  DATA_ROOT="${SCRIPT_DIR}"
fi
RESULTS_DIR="${DATA_ROOT}/results"
GEN_DEVICE="${GEN_DEVICE:-cuda:0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

case "${MODE}" in
  smoke)
    mkdir -p "${RESULTS_DIR}"
    python3 "${SCRIPT_DIR}/00_validate_inputs.py" --issue-limit 1 --template-count 2 --max-cues 3 --repeats 1
    python3 "${SCRIPT_DIR}/01_build_prompts.py" --mode smoke
    echo "Smoke prompts are ready at ${RESULTS_DIR}/prompts_smoke.csv"
    echo "Run generation next with: python3 ${SCRIPT_DIR}/02_run_generation.py --prompts ${RESULTS_DIR}/prompts_smoke.csv --out-jsonl ${RESULTS_DIR}/generations_smoke.jsonl --out-csv ${RESULTS_DIR}/generations_smoke.csv --device ${GEN_DEVICE} --batch-size ${GEN_BATCH_SIZE}"
    ;;
  pilot)
    mkdir -p "${RESULTS_DIR}"
    python3 "${SCRIPT_DIR}/00_validate_inputs.py"
    python3 "${SCRIPT_DIR}/01_build_prompts.py" --mode pilot
    python3 "${SCRIPT_DIR}/02_run_generation.py" --prompts "${RESULTS_DIR}/prompts_pilot.csv" --out-jsonl "${RESULTS_DIR}/generations_pilot.jsonl" --out-csv "${RESULTS_DIR}/generations_pilot.csv" --device "${GEN_DEVICE}" --batch-size "${GEN_BATCH_SIZE}"
    python3 "${SCRIPT_DIR}/03_run_stance_eval.py" --generations "${RESULTS_DIR}/generations_pilot.csv" --out-jsonl "${RESULTS_DIR}/evaluated_pilot.jsonl" --out-csv "${RESULTS_DIR}/evaluated_pilot.csv" --device "${EVAL_DEVICE}" --batch-size "${EVAL_BATCH_SIZE}"
    python3 "${SCRIPT_DIR}/04_analyse_prelim.py" --evaluated "${RESULTS_DIR}/evaluated_pilot.csv" --out-dir "${RESULTS_DIR}/analysis_pilot"
    ;;
  analyse)
    python3 "${SCRIPT_DIR}/04_analyse_prelim.py" --evaluated "${RESULTS_DIR}/evaluated_pilot.csv" --out-dir "${RESULTS_DIR}/analysis_pilot"
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|analyse}" >&2
    exit 2
    ;;
esac
