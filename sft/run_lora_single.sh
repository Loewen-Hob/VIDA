#!/bin/bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcudnn.so.9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=7
unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK

MODEL_PATH="/VIDA/Qwen3-VL-8B-Instruct"
DATA_FILE="../datasets/MS-S-train_sft.jsonl"
OUTPUT_DIR="output/qwen3_vl_8b_lora_sft_stable"

swift sft \
    --model "${MODEL_PATH}" \
    --dataset "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit true \
    \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --max_pixels 1003520 \
    \
    --gradient_checkpointing true \
    --save_steps 50 \
    --eval_steps 50 \
    --save_only_model true \
    --save_total_limit 3 \
    --logging_steps 5 \
    --lazy_tokenize true \
    --dataloader_num_workers 1