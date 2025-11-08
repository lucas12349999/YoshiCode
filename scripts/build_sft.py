import json, sys
from utils import ensure_dirs

# usage: python scripts/build_sft.py src1.jsonl src2.jsonl ... > data/processed/sft.jsonl
ensure_dirs()
sources = sys.argv[1:]
for s in sources:
    for line in open(s):
        ex = json.loads(line)
        inst = ex.get("instruction") or ex.get("prompt")
        outp = ex.get("output") or ex.get("response")
        if inst and outp:
            print(json.dumps({"instruction": inst, "input": "", "output": outp}))
