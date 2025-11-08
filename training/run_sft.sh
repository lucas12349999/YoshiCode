#!/usr/bin/env bash
set -euo pipefail
IN="${1:-out/yoshicode-pretrain}"
SFT_JSONL="${2:-data/processed/sft.jsonl}"
OUT="${3:-out/yoshicode-sft}"

export WANDB_PROJECT="${WANDB_PROJECT:-yoshicode-sft}"

accelerate launch --config_file training/accelerate_config.yaml scripts/run_sft.py \
  "$IN" "$SFT_JSONL" "$OUT" \
  --lr "${LR:-5e-5}" \
  --epochs "${EPOCHS:-3}" \
  --bsz "${BSZ:-2}" \
  --accum "${ACCUM:-16}" \
  --fp16 ${LORA_FLAG:-}
