#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${1:-out/yoshicode-sft}"
PORT="${2:-8000}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --host 0.0.0.0 --port "${PORT}" \
  --max-model-len 2048 \
  --served-model-name yoshicode-1b \
  --chat-template serving/chat_template.yaml
