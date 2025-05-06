#!/bin/bash
set -e

# ========== 用户可配置变量 ==========
STAGE=1  # 控制从哪个阶段开始执行，合法值：1~5

OUTPUT_DIR="/root/autodl-tmp/V429/exp_dense_batch_8"
DATA_PATH="/root/V429/dataset/AskNews-NER-v0"
MODEL_CONFIG="/root/V429/hf_models/TinyLlama/TinyLlama-1.1B-Chat-v1.0/config.json"
ZERO_TO_FP32="${OUTPUT_DIR}/zero_to_fp32.py"
PROJECT_NAME="llama-training-20250504-ltx-exp-moe-batch-8"
WANDB_ENTITY="6998gp_TLA"

# ========== 阶段 1：训练 ==========
if [ "$STAGE" -le 1 ]; then
  echo "🚀 Starting DeepSpeed training..."
  deepspeed --num_gpus 2 /root/V429/project_run.py \
    --run_mode train \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --model_max_length 1024 \
    --logging_steps 5 \
    --save_interval 500 \
    --bf16 \
    --use_wandb true \
    --wandb_project "$PROJECT_NAME" \
    --wandb_entity "$WANDB_ENTITY" \
    --deepspeed /root/V429/ds_config.json \
    --experiment_type moe \
    --router_strategy random \
    --data_path "$DATA_PATH" \
    --use_lora false

  echo "✅ Training completed."
fi

# ========== 阶段 2：准备 Checkpoint 结构 ==========
if [ "$STAGE" -le 2 ]; then
  echo "🔍 Searching best checkpoint under: ${OUTPUT_DIR} ..."
  CKPT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "epoch*_best" | head -n 1)

  if [ -z "$CKPT_DIR" ]; then
    echo "❌ No checkpoint found under ${OUTPUT_DIR}, exiting."
    exit 1
  fi

  echo "✅ Found best checkpoint directory: $CKPT_DIR"
  echo "🛠️ Preparing DeepSpeed-style checkpoint..."

  if [ ! -d "${CKPT_DIR}/global_step0" ]; then
    echo "📁 Creating global_step0/ directory..."
    mkdir -p "${CKPT_DIR}/global_step0"
    mv "${CKPT_DIR}/mp_rank_00_model_states.pt" "${CKPT_DIR}/global_step0/" || true
    mv "${CKPT_DIR}"/bf16_zero_pp_rank_* "${CKPT_DIR}/global_step0/" || true
  fi

  echo "📝 Writing latest checkpoint info..."
  echo "global_step0" > "${CKPT_DIR}/latest"
fi

# ========== 阶段 3：转换为 HuggingFace 权重 ==========
if [ "$STAGE" -le 3 ]; then
  echo "🔄 Running zero_to_fp32 to create pytorch_model.bin..."
  python "${ZERO_TO_FP32}" "${CKPT_DIR}" "${OUTPUT_DIR}/pytorch_model.bin"

  # === 特殊处理：pytorch_model.bin 是目录的情况 ===
  if [ -d "${OUTPUT_DIR}/pytorch_model.bin" ]; then
    echo "📦 Flattening pytorch_model.bin folder..."

    TEMP_DIR="${OUTPUT_DIR}/pytorch_model_temp"
    mv "${OUTPUT_DIR}/pytorch_model.bin" "$TEMP_DIR"
    mv "${TEMP_DIR}/pytorch_model.bin" "${OUTPUT_DIR}/"
    rm -rf "$TEMP_DIR"

    echo "✅ Flattened model file moved to ${OUTPUT_DIR}/pytorch_model.bin"
  fi

  echo "✅ Model conversion complete."
fi

# ========== 阶段 4：复制配置和 tokenizer ==========
if [ "$STAGE" -le 4 ]; then
  echo "📋 Copying tokenizer and config files..."
  cp "${MODEL_CONFIG}" "${OUTPUT_DIR}/config.json"
  cp "${CKPT_DIR}/tokenizer.model" "${OUTPUT_DIR}/" || true
  cp "${CKPT_DIR}/tokenizer_config.json" "${OUTPUT_DIR}/" || true
  cp "${CKPT_DIR}/special_tokens_map.json" "${OUTPUT_DIR}/" || true

  echo "✅ Files prepared in: ${OUTPUT_DIR}"
fi

# ========== 阶段 5：评估模型 ==========
if [ "$STAGE" -le 5 ]; then
  echo "🧪 Running evaluation..."
  python /root/V429/eval.py \
    --model_dir "$OUTPUT_DIR" \
    --input_jsonl "${DATA_PATH}.jsonl" \
    --output_jsonl "${OUTPUT_DIR}/eval/outputs.jsonl"

  echo "✅ Evaluation complete. Results saved to: ${OUTPUT_DIR}/eval/outputs.jsonl"
fi
