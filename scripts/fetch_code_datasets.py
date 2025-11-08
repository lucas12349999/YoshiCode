import json
from datasets import load_dataset
from utils import ensure_dirs

ensure_dirs()
ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
limit = 50000   # small pilot to dry-run locally; increase on cloud
out = open("data/processed/pretrain-train.jsonl","w")
cnt = 0
for ex in ds:
    text = ex.get("content") or ex.get("code") or ""
    if text:
        out.write(json.dumps({"text": text})+"\n")
        cnt += 1
        if cnt >= limit: break
out.close()
print(f"wrote {cnt} records → data/processed/pretrain-train.jsonl")
