#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-outputs/qwen3_vl_8b_sft_merged}
DATA_FILE=${DATA_FILE:-data/MS-S-train_rl.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/vida_gspo_rl}
PROMPT_FILE=${PROMPT_FILE:-outputs/system_prompt.txt}
VISUAL_REWARD=${VISUAL_REWARD:-lexical}
REWARD_MODEL=${REWARD_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
RLHF_TYPE=${RLHF_TYPE:-grpo}

mkdir -p "$(dirname "${PROMPT_FILE}")"
python - <<'PY' > "${PROMPT_FILE}"
from vida.constants import SYSTEM_PROMPT
print(SYSTEM_PROMPT)
PY

export VIDA_REWARD_MODEL="${REWARD_MODEL}"
export VIDA_VISUAL_REWARD="${VISUAL_REWARD}"

swift rlhf \
  --rlhf_type "${RLHF_TYPE}" \
  --model_type qwen3_vl \
  --model "${MODEL_PATH}" \
  --train_type lora \
  --dataset "${DATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_funcs vida_gspo_reward \
  --external_plugins vida/reward.py \
  --num_train_epochs 1 \
  --num_generations 4 \
  --max_completion_length 128 \
  --beta 0.01 \
  --lora_rank 64 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 50 \
  --save_only_model true \
  --report_to tensorboard \
  --system "${PROMPT_FILE}" \
  --deepspeed zero2
