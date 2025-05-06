#!/bin/bash
set -e

DATA_PATH="/root/V429/dataset/AskNews-NER-v0"
MODEL_CONFIG="/root/V429/hf_models/TinyLlama/TinyLlama-1.1B-Chat-v1.0/config.json"
WANDB_ENTITY="6998gp_TLA"

run_training() {
  OUTPUT_DIR=$1
  NUM_GPUS=$2
  EXPERIMENT_TYPE=$3
  ROUTER_STRATEGY=$4
  PROJECT_NAME=$5

  echo "🚀 Starting training: $PROJECT_NAME"
  deepspeed --num_gpus "$NUM_GPUS" /root/V429/project_run.py \
    --run_mode train \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --model_max_length 1024 \
    --logging_steps 5 \
    --save_interval 500 \
    --bf16 \
    --use_wandb true \
    --wandb_project "$PROJECT_NAME" \
    --wandb_entity "$WANDB_ENTITY" \
    --deepspeed /root/V429/ds_config.json \
    --experiment_type "$EXPERIMENT_TYPE" \
    --router_strategy "$ROUTER_STRATEGY" \
    --data_path "$DATA_PATH" \
    --use_lora false

  echo "✅ Training complete for $PROJECT_NAME"
}

# 模型 1：原始设置（moe，2卡）
#run_training "/root/autodl-tmp/V429/exp_moe_batch_8" 2 "moe" "random" "llama-training-20250504-ltx-exp-moe-batch-8"

# 模型 2：dense，1卡
#run_training "/root/autodl-tmp/V429/exp_dense_batch_8" 1 "dense" "none" "llama-training-20250504-ltx-exp-dense1gpu-batch-8"

# 模型 3：moe，1卡 OOM
#run_training "/root/autodl-tmp/V429/exp_moe_1gpu_batch_4" 1 "moe" "random" "llama-training-20250504-ltx-exp-moe1gpu-batch-4"
