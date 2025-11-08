# YoshiCode-1B

Compact 1B-param code assistant. TinyLlama intermediate (1.1B) base → code continual pretraining → instruction SFT → vLLM serving.

## Local prep (no GPU)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/pull_base.py
python scripts/fetch_code_datasets.py \
  --datasets codeparrot/codeparrot-clean codeparrot/github-code code_search_net:python \
  --samples-per-dataset 1000 \
  --train-out data/processed/pretrain-train.jsonl \
  --val-out data/processed/pretrain-val.jsonl \
  --license-manifest data/processed/licenses.csv
# tweak --samples-per-dataset upward on GPU nodes
python -m compileall scripts
```
The command above streams cleaned OSS code from CodeParrot, GitHub-Code (Python filtered), and CodeSearchNet (Python subset) with per-source limits for quick local validation. Update `SOURCES.md` with URLs and timestamps after each pull.
