import math, json, torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# usage: python scripts/quick_perplexity.py <model_dir> <holdout.jsonl>
model_dir = sys.argv[1]
holdout   = sys.argv[2]

tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_dir).eval()

losses = []
for i,line in enumerate(open(holdout)):
    text = json.loads(line)["text"]
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    losses.append(out.loss.item())
ppl = math.exp(sum(losses)/len(losses))
print(f"perplexity: {ppl:.2f} over {len(losses)} samples")
