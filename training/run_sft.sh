#!/usr/bin/env bash
set -euo pipefail

MODEL_IN="${1:-out/yoshicode-pretrain}"
TRAIN_FILE="${2:-data/processed/sft.jsonl}"
EVAL_FILE="${3:-data/eval/sft_eval.jsonl}"
OUTPUT_DIR="${4:-out/yoshicode-sft}"

accelerate launch scripts/run_sft.py \
  --model-name "$MODEL_IN" \
  --train-file "$TRAIN_FILE" \
  --eval-file "$EVAL_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 3e-5 \
  --num-train-epochs 3 \
  --warmup-ratio 0.03 \
  --max-seq-length 2048 \
  --deepspeed training/deepspeed_zero3.json \
  --logging-steps 50 \
  --save-steps 500 \
  --eval-steps 500 \
  --report-to tensorboard \
  "$@"
