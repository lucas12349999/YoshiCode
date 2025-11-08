from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction fine-tuning launcher for YoshiCode-1B.")
    parser.add_argument("--model-name", default="out/yoshicode-pretrain", help="Base model directory or HF repo.")
    parser.add_argument("--tokenizer-name", default=None, help="Optional tokenizer directory (defaults to model).")
    parser.add_argument("--train-file", default="data/processed/sft.jsonl")
    parser.add_argument("--eval-file", default="data/eval/sft_eval.jsonl")
    parser.add_argument("--output-dir", default="out/yoshicode-sft")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing for higher throughput.")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision where available.")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    parser.add_argument("--deepspeed", default="training/deepspeed_zero3.json")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--report-to", nargs="*", default=["tensorboard"])
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    return parser.parse_args()


def load_sft_dataset(train_path: str, eval_path: Optional[str], num_proc: int) -> Dict[str, Dataset]:
    data_files = {"train": train_path}
    if eval_path and Path(eval_path).is_file():
        data_files["eval"] = eval_path
    ds = load_dataset("json", data_files=data_files)
    if "train" not in ds:
        raise ValueError("Training JSONL not found or empty.")
    if "instruction" not in ds["train"].column_names:
        raise ValueError("SFT JSONL must contain instruction/input/output fields.")
    if num_proc and num_proc > 1:
        ds = ds.map(lambda x: x, num_proc=num_proc)
    return ds


def build_formatting_func() -> Callable[[Dict], List[str]]:
    def format_example(example: Dict) -> List[str]:
        inst = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        out = example.get("output", "").strip()
        prompt = "### Instruction:\n" + inst
        if inp:
            prompt += "\n\n### Input:\n" + inp
        prompt += "\n\n### Response:\n"
        text = prompt + out
        return [text]

    return format_example


def maybe_build_lora_config(args: argparse.Namespace) -> Optional[LoraConfig]:
    if not args.use_lora:
        return None
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if "wandb" not in args.report_to:
            args.report_to.append("wandb")

    dataset = load_sft_dataset(args.train_file, args.eval_file, args.dataset_num_proc)
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if "eval" in dataset and args.max_eval_samples:
        dataset["eval"] = dataset["eval"].select(range(min(args.max_eval_samples, len(dataset["eval"]))))

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    peft_config = maybe_build_lora_config(args)
    formatting_func = build_formatting_func()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if "eval" in dataset else "no",
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if peft_config:
        trainer.model.config.save_pretrained(Path(args.output_dir) / "adapter_config")
    print(f"[run_sft] saved model → {args.output_dir}")


if __name__ == "__main__":
    main()
