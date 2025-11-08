# YoshiCode-1B
1B-param code assistant (TinyLlama base) — continual pretraining + SFT, optional vLLM serving.

## Environment
```bash
conda create -n yoshicode python=3.11 -y
conda activate yoshicode
pip install -r requirements.txt
accelerate config
```

## Data pipeline
1. **Pull base model (optional local cache)**
   ```bash
   python scripts/pull_base.py
   ```
2. **Code-only corpora for continual pretraining**
   ```bash
   python scripts/fetch_code_datasets.py \
     --datasets all \
     --languages python \
     --limit-per-dataset 200000 \
     --train-output data/processed/pretrain-train.jsonl \
     --eval-output data/processed/pretrain-eval.jsonl
   ```
   This streams from `codeparrot/codeparrot-clean`, `bigcode/stack-dedup`, and `code_search_net` (Python config) with deterministic train/eval splits. Update `SOURCES.md` with dataset URLs + timestamps after each run.
3. **StackOverflow Q&A for instruction tuning**
   ```bash
   export STACKEXCHANGE_KEY=<optional_key>
   python scripts/fetch_stackoverflow.py \
     --tag python \
     --pages 25 \
     --page-size 50 \
     --output data/raw/stackoverflow_python.jsonl
   ```
4. **Assemble SFT set (CodeAlpaca/OpenHermes + StackOverflow)**
   ```bash
   python scripts/build_sft.py data/raw/codealpaca.jsonl data/raw/openhermes.jsonl \
     --stackoverflow data/raw/stackoverflow_python.jsonl \
     --output data/processed/sft.jsonl \
     --include-metadata
   ```
5. **Create evaluation splits**
   ```bash
   python scripts/build_holdout.py data/processed/pretrain-train.jsonl \
     --size 5000 --output data/processed/pretrain-eval.jsonl
   python scripts/build_holdout.py data/processed/sft.jsonl \
     --size 1000 --output data/eval/sft_eval.jsonl
   ```

## Training (cloud GPU)
1. Copy `data/processed/pretrain-*.jsonl` and `out/base` to the GPU node.
2. **Continual pretraining**
   ```bash
   bash training/run_pretrain.sh out/base out/base \
     data/processed/pretrain-train.jsonl data/processed/pretrain-eval.jsonl \
     out/yoshicode-pretrain --num-train-epochs 1 --save-steps 1000
   ```
   Pass additional overrides (e.g., `--learning-rate`, `--max-steps`, `--wandb-project`) after the defaults above.
3. **Instruction tuning**
   ```bash
   bash training/run_sft.sh out/yoshicode-pretrain data/processed/sft.jsonl \
     data/eval/sft_eval.jsonl out/yoshicode-sft \
     --num-train-epochs 4 --learning-rate 4e-5 --use-lora --lora-rank 128
   ```
   Append/override CLI flags to enable LoRA, adjust batch sizes, log to Weights & Biases, etc.

## Evaluation & serving
- Generate quick perplexity on code hold-out:
  ```bash
  python scripts/quick_perplexity.py --model out/yoshicode-pretrain --holdout data/processed/pretrain-eval.jsonl
  ```
- Run instruction-suite smoke tests (optionally execute generated Python):
  ```bash
  python scripts/run_eval_suite.py --model out/yoshicode-sft --execute-python
  ```
- Launch vLLM after SFT if desired:
  ```bash
  bash serving/serve_vllm.sh out/yoshicode-sft 8000
  python scripts/test_api.py
  ```

Keep `SOURCES.md` updated with every dataset/API pull (include URLs, access dates, and attribution strings).
