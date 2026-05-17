#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
DATA_FILE=${DATA_FILE:-data/MS-S-train_sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3_vl_8b_sft}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29600}
TRAIN_TYPE=${TRAIN_TYPE:-lora}

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  -m swift.cli.sft \
  --model_type qwen3_vl \
  --model "${MODEL_ID}" \
  --dataset "${DATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_type "${TRAIN_TYPE}" \
  --lora_rank 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --max_length 2048 \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --gradient_checkpointing true \
  --save_steps 200 \
  --eval_steps 200 \
  --save_only_model true \
  --logging_steps 10 \
  --lazy_tokenize true \
  --dataloader_num_workers 1
