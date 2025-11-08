from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from utils import ensure_dirs

ensure_dirs()
base = "tinyllama/TinyLlama-1.1B"
Path("out/base").mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(base, use_fast=True)
tok.save_pretrained("out/base")
model = AutoModelForCausalLM.from_pretrained(base)
model.save_pretrained("out/base")
print("Saved base → out/base")
