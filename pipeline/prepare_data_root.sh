#!/usr/bin/env bash
# Create the scratch/output root for generated pipeline artifacts.
# Committed inputs (data/input, data/reference) are read directly from the repo,
# so they do not need to be copied here.
set -euo pipefail

OUTPUT_ROOT="${THESIS_PIPELINE_DATA_ROOT:-/data/kell8360/thesis_framing_pipeline}"

mkdir -p "${OUTPUT_ROOT}/results"

echo "Prepared output root: ${OUTPUT_ROOT}"
echo "Results: ${OUTPUT_ROOT}/results"
