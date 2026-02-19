#!/bin/bash

cat > ds_config_lora.json <<EOF
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": "auto"
  }
}
EOF

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MODEL_ID="Qwen3-VL-8B-Instruct"
# 你的数据路径
DATA_FILE="../datasets/MS-S-train_sft.jsonl"
OUTPUT_DIR="output/qwen3_vl_8b_lora_sft"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29600 \
    -m swift.cli.sft \
    --model_type qwen3_vl \
    --model ${MODEL_ID} \
    --dataset ${DATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit true \
    \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_length 2048 \
    \
    --deepspeed ds_config_lora.json \
    --gradient_checkpointing true \
    \
    --save_steps 200 \
    --eval_steps 200 \
    --save_only_model true \
    --logging_steps 10 \
    --lazy_tokenize true \
    --dataloader_num_workers 1