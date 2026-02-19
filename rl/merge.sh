#!/bin/bash
SFT_CKPT_DIR="/sft/output/qwen3_vl_8b_lora_sft_stable/v3-20251228-102420/checkpoint-144"

MERGED_DIR="/sft/output/qwen3_vl_8b_sft_merged"

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --ckpt_dir "${SFT_CKPT_DIR}" \
    --merge_lora true \
    --output_dir "${MERGED_DIR}"