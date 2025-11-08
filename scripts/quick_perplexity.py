from __future__ import annotations

import argparse
import json
import math
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perplexity over a JSONL holdout set.")
    parser.add_argument("--model", default="out/yoshicode-pretrain")
    parser.add_argument("--holdout", default="data/processed/pretrain-eval.jsonl")
    parser.add_argument("--max-samples", type=int, default=200, help="Cap the number of examples evaluated.")
    parser.add_argument("--device", default=None, help="Torch device override (e.g., cuda:0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    if args.device:
        model.to(args.device)

    losses: List[float] = []
    with open(args.holdout, encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if args.max_samples and idx >= args.max_samples:
                break
            text = json.loads(line)["text"]
            inputs = tokenizer(text, return_tensors="pt")
            if args.device:
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
    ppl = math.exp(sum(losses) / len(losses))
    print(f"[quick_perplexity] perplexity={ppl:.2f} over {len(losses)} samples")


if __name__ == "__main__":
    main()
