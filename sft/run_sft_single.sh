
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export DEEPSPEED_DISTRIBUTED_BACKEND=nccl


cat > ds_config_final.json <<EOF
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": "auto"
  }
}
EOF

MODEL_PATH="/root/paddlejob/workspace/env_run/output/VIDA/Qwen3-VL-8B-Instruct"
DATA_FILE="../datasets/MS-S-train_sft.jsonl"
OUTPUT_DIR="output/qwen3_vl_8b_full_sft_final"

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model "${MODEL_PATH}" \
    --train_type full \
    --dataset "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    \
    --deepspeed ds_config_final.json \
    --gradient_checkpointing true \
    \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --save_only_model true \
    --lazy_tokenize true \
    --dataloader_num_workers 1

    