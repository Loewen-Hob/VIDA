#!/usr/bin/env bash
set -euo pipefail

GROUND_TRUTH_FILE=${GROUND_TRUTH_FILE:?Set GROUND_TRUTH_FILE to a ground-truth JSONL file.}
PREDICTION_FILE=${PREDICTION_FILE:?Set PREDICTION_FILE to a prediction JSONL file.}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}

python -m vida.evaluate \
  --ground-truth-file "${GROUND_TRUTH_FILE}" \
  --prediction-file "${PREDICTION_FILE}" \
  --embedding-model "${EMBEDDING_MODEL}"
