## Pretraining corpora
- `codeparrot/codeparrot-clean` — Apache-2.0 — https://huggingface.co/datasets/codeparrot/codeparrot-clean — pulled with `scripts/fetch_code_datasets.py` (record run date + commit hash).
- `bigcode/stack-dedup` (Python) — MIT — https://huggingface.co/datasets/bigcode/stack-dedup.
- `code_search_net` (Python) — Apache-2.0 — https://huggingface.co/datasets/code_search_net.

## Instruction/SFT corpora
- `CodeAlpaca-20k` — Apache-2.0 — https://github.com/sahil280114/codealpaca.
- `OpenHermes` code subset — CC BY-SA 4.0 — https://huggingface.co/datasets/teknium/OpenHermes-2.5.
- **StackOverflow Q&A (Python tag)** — CC BY-SA 4.0 — collected via StackExchange API (`scripts/fetch_stackoverflow.py`). Store JSONL under `data/raw/stackoverflow_*.jsonl` and keep the download timestamp + tag query.

> Record access date/time, command used, and any filtering performed whenever you refresh a dataset.
