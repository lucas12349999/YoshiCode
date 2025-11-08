#!/usr/bin/env bash
set -euo pipefail
python scripts/run_sft.py out/yoshicode-pretrain data/processed/sft.jsonl out/yoshicode-sft
