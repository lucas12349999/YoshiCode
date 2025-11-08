import sys
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# usage: python scripts/run_sft.py <model_dir_in> <sft.jsonl> <model_dir_out>
model_dir_in  = sys.argv[1]
sft_jsonl     = sys.argv[2]
model_dir_out = sys.argv[3]

model = AutoModelForCausalLM.from_pretrained(model_dir_in, torch_dtype="auto")
tok   = AutoTokenizer.from_pretrained(model_dir_in, use_fast=True)

train = load_dataset("json", data_files={"train": sft_jsonl})["train"]

trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=train,
    max_seq_length=2048,
    packing=False,
)

trainer.train()
trainer.save_model(model_dir_out)
tok.save_pretrained(model_dir_out)
print(f"Saved SFT model → {model_dir_out}")
