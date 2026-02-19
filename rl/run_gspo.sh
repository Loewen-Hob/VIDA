#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

PROMPT_FILE="system_prompt.txt"
cat <<EOF > ${PROMPT_FILE}
You are a professional interior designer. Based on the images and requirements, identify the user's latent needs and formulate appropriate questions.
Please understand which element should be the primary focus of your inquiry, then pose only one question at a time. Avoid asking multiple questions simultaneously. Keep your word count within the limits of a single question, ensuring it remains concise and not overly complex.
EOF

MODEL_PATH="/root/paddlejob/workspace/env_run/output/VIDA/sft/output/qwen3_vl_8b_sft_merged"

DATA_FILE="/root/paddlejob/workspace/env_run/output/VIDA/datasets/MS-S-train_rl.jsonl"
OUTPUT_DIR="output/vida_gspo_rl"


swift rlhf \
    --rlhf_type grpo \
    --model_type qwen3_vl \
    --model "$MODEL_PATH" \
    --train_type lora \
    --dataset "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    \
    --reward_funcs vida_gspo_reward \
    --external_plugins custom_reward_gspo.py \
    \
    --num_train_epochs 1 \
    --num_generations 4 \
    --max_completion_length 128 \
    --beta 0.01 \
    \
    --lora_rank 64 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_only_model true \
    --report_to tensorboard \
    --system "${PROMPT_FILE}" \
    --deepspeed zero2