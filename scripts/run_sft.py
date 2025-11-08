#!/usr/bin/env python3
"""
Usage:
  python scripts/run_sft.py <model_dir_in> <sft.jsonl> <model_dir_out> \
    --lr 5e-5 --epochs 3 --bsz 2 --accum 16 --fp16 \
    --lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
"""
import sys, argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_in")
    ap.add_argument("sft_jsonl")
    ap.add_argument("model_out")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--wandb-project", default="yoshicode-sft")
    return ap.parse_args()

def main():
    args = parse()
    tok = AutoTokenizer.from_pretrained(args.model_in, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_in, torch_dtype="auto")

    if args.lora:
        assert PEFT_AVAILABLE, "peft not installed"
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lcfg)

    ds = load_dataset("json", data_files={"train": args.sft_jsonl})["train"]

    targs = TrainingArguments(
        output_dir=args.model_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        report_to=["wandb"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=targs,
        train_dataset=ds,
        max_seq_length=args.max_len,
        packing=False,
    )
    trainer.train()
    trainer.save_model(args.model_out)
    tok.save_pretrained(args.model_out)
    print(f"[OK] saved → {args.model_out}")

if __name__ == "__main__":
    sys.exit(main())
