#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-out/base}"
TOKENIZER="${2:-out/base}"
TRAIN_FILE="${3:-data/processed/pretrain-train.jsonl}"
EVAL_FILE="${4:-data/processed/pretrain-eval.jsonl}"
OUTPUT_DIR="${5:-out/yoshicode-pretrain}"

accelerate launch training/run_pretrain.py \
  --model-name "$MODEL" \
  --tokenizer-name "$TOKENIZER" \
  --train-file "$TRAIN_FILE" \
  --eval-file "$EVAL_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --block-size 2048 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --num-train-epochs 1 \
  --warmup-ratio 0.02 \
  --weight-decay 0.1 \
  --save-steps 1000 \
  --eval-steps 1000 \
  --logging-steps 50 \
  --deepspeed training/deepspeed_zero3.json \
  --report-to tensorboard \
  "$@"
