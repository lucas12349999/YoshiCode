from pathlib import Path

def ensure_dirs():
    for p in ["data/raw","data/processed","data/eval","out"]:
        Path(p).mkdir(parents=True, exist_ok=True)
