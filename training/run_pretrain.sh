#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-out/base}"
TOKENIZER="${2:-out/base}"
TRAIN_FILE="${3:-data/processed/pretrain-train.jsonl}"

accelerate launch -m transformers.examples.pytorch.language_modeling.run_clm \
  --model_name_or_path "$MODEL" \
  --tokenizer_name "$TOKENIZER" \
  --train_file "$TRAIN_FILE" \
  --block_size 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --fp16 --gradient_checkpointing \
  --deepspeed training/deepspeed_zero3.json \
  --save_steps 1000 --logging_steps 50 \
  --output_dir out/yoshicode-pretrain

  