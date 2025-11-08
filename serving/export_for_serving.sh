#!/usr/bin/env bash
set -euo pipefail
IN="${1:-out/yoshicode-sft}"
OUT="${2:-serving/export}"
TEMPLATE="${3:-serving/chat_template.yaml}"

mkdir -p "$OUT"
cp -r "$IN"/* "$OUT"/
cp "$TEMPLATE" "$OUT"/chat_template.yaml
echo "[OK] packaged model + tokenizer + chat_template → $OUT"
