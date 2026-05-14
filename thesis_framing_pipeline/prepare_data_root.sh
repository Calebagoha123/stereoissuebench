#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${THESIS_PIPELINE_DATA_ROOT:-/data/kell8360/thesis_framing_pipeline}"

mkdir -p "${DATA_ROOT}/input_data" "${DATA_ROOT}/results"

cp -f "${SCRIPT_DIR}/input_data/"*.csv "${DATA_ROOT}/input_data/"
cp -f "${SCRIPT_DIR}/results/prompts_smoke.csv" "${DATA_ROOT}/results/"
cp -f "${SCRIPT_DIR}/results/prompts_pilot.csv" "${DATA_ROOT}/results/"

echo "Prepared data root: ${DATA_ROOT}"
echo "Inputs: ${DATA_ROOT}/input_data"
echo "Results: ${DATA_ROOT}/results"
