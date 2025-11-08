#!/usr/bin/env bash
set -euo pipefail

MODEL_IN="${1:-out/base}"
TOKENIZER_IN="${2:-out/base}"
TRAIN_FILE="${3:-data/processed/pretrain-train.jsonl}"
VAL_FILE="${4:-data/processed/pretrain-eval.jsonl}"
OUT_DIR="${5:-out/yoshicode-pretrain}"

shift 5 || true

accelerate launch --config_file training/accelerate_config.yaml training/run_pretrain.py \
  --model-name "$MODEL_IN" \
  --tokenizer-name "$TOKENIZER_IN" \
  --train-file "$TRAIN_FILE" \
  --eval-file "$VAL_FILE" \
  --output-dir "$OUT_DIR" \
  --block-size 2048 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --num-train-epochs 1 \
  --weight-decay 0.1 \
  --warmup-ratio 0.02 \
  --logging-steps 20 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --report-to tensorboard \
  "$@"
