#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?Set MODEL to a model path or hub id.}
ADAPTER=${ADAPTER:-}
INPUT_FILE=${INPUT_FILE:?Set INPUT_FILE to an inference JSONL file.}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/predictions.jsonl}

if [[ -n "${ADAPTER}" ]]; then
  python -m vida.infer \
    --model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}"
else
  python -m vida.infer \
    --model "${MODEL}" \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}"
fi
