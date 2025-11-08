from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continual pretraining launcher for YoshiCode-1B.")
    parser.add_argument("--model-name", default="out/base", help="Pretrained model directory or HF ID.")
    parser.add_argument("--tokenizer-name", default="out/base", help="Tokenizer directory or HF ID.")
    parser.add_argument("--train-file", default="data/processed/pretrain-train.jsonl")
    parser.add_argument("--eval-file", default="data/processed/pretrain-eval.jsonl")
    parser.add_argument("--output-dir", default="out/yoshicode-pretrain")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use -1 to disable max-steps override.")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--flash-attention-impl", default="flash_attention_2", help="HF attention implementation.")
    parser.add_argument("--bf16", action="store_true", help="Train in bf16 precision where supported.")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    parser.add_argument("--deepspeed", default="training/deepspeed_zero3.json")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--report-to", nargs="*", default=["tensorboard"])
    parser.add_argument("--wandb-project", default=None, help="Optional WANDB project (auto-adds to report_to).")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    return parser.parse_args()


def load_corpora(train_file: str, eval_file: Optional[str]) -> DatasetDict:
    data_files: Dict[str, str] = {"train": train_file}
    if eval_file and Path(eval_file).is_file():
        data_files["validation"] = eval_file
    dataset = load_dataset("json", data_files=data_files)
    missing_cols = [split for split in dataset if "text" not in dataset[split].column_names]
    if missing_cols:
        raise ValueError(f"JSONL must contain a 'text' field for splits: {missing_cols}")
    return dataset


def tokenize_corpus(dataset: DatasetDict, tokenizer, block_size: int, num_workers: int, max_train=None, max_eval=None):
    column_names = dataset["train"].column_names

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers or None,
        remove_columns=column_names,
        desc="Tokenizing text",
    )

    def group_texts(examples):
        concatenated: List[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = len(concatenated)
        total_length = total_length - (total_length % block_size)
        sequences = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        result = {
            "input_ids": sequences,
            "attention_mask": [[1] * block_size for _ in range(len(sequences))],
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_workers or None,
        desc=f"Grouping texts into chunks of {block_size}",
    )

    if max_train:
        tokenized["train"] = tokenized["train"].select(range(min(max_train, len(tokenized["train"]))))
    if "validation" in tokenized and max_eval:
        tokenized["validation"] = tokenized["validation"].select(range(min(max_eval, len(tokenized["validation"]))))
    return tokenized


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.wandb_project:
        args.report_to = list(set([*args.report_to, "wandb"]))
        os.environ["WANDB_PROJECT"] = args.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = None if args.flash_attention_impl in {"none", "auto"} else args.flash_attention_impl
    model_kwargs = {}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except ValueError as exc:
        if attn_impl:
            print(f"[run_pretrain] FlashAttention load failed ({exc}); retrying with default attention.")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        else:
            raise
    dataset = load_corpora(args.train_file, args.eval_file)
    tokenized = tokenize_corpus(
        dataset,
        tokenizer,
        args.block_size,
        args.preprocessing_num_workers,
        args.max_train_samples,
        args.max_eval_samples,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if "validation" in tokenized else "no",
        save_strategy="steps",
        report_to=args.report_to,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        save_total_limit=args.save_total_limit,
        deepspeed=args.deepspeed,
        logging_dir=os.path.join(args.output_dir, "logs"),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    resume = args.resume_from
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
