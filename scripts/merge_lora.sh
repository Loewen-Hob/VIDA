#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?Set CKPT_DIR to the LoRA checkpoint directory.}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/merged_model}

swift export \
  --ckpt_dir "${CKPT_DIR}" \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"
