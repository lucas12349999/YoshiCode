#!/usr/bin/env python3
import argparse, csv, json, random, sys, os
from pathlib import Path
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH

LICENSE_HINTS = {
  "codeparrot/codeparrot-clean": "Apache-2.0 (cleaned OSS code)",
  "codeparrot/github-code": "Public GitHub repos (mixed permissive licenses)",
  "code_search_net": "Varies by source repo; prefer permissive subsets",
  "bigcode/stackoverflow-python": "CC BY-SA 4.0 (requires attribution + SA)"
}

def stream(name: str):
    if ":" in name:
        dset, subset = name.split(":", 1)
        ds = load_dataset(dset, subset, split="train", streaming=True, trust_remote_code=True)
    else:
        ds = load_dataset(name, split="train", streaming=True, trust_remote_code=True)
    if name.startswith("codeparrot/github-code"):
        ds = ds.filter(lambda ex: (ex.get("language") or "").lower() == "python")
    return ds

def minhash(s: str, num_perm=64):
    m = MinHash(num_perm=num_perm)
    for tok in s.split():
        m.update(tok.encode("utf-8"))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--samples-per-dataset", type=int, default=500_000)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--max-bytes", type=int, default=20000)
    ap.add_argument("--license-manifest", required=True)
    ap.add_argument("--shard-size", type=int, default=1_000_000)  # bytes per shard file (approx)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    Path(Path(args.train_out).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(args.val_out).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(args.license_manifest).parent).mkdir(parents=True, exist_ok=True)

    # lightweight dedupe with LSH over MinHash
    lsh = MinHashLSH(threshold=0.9, num_perm=64)

    def write_sharded(path_base):
        # rotate files to avoid huge single jsonl
        idx, f, bytes_written = 0, None, 0
        while True:
            yield idx, f, bytes_written
            idx += 1

    train_f = open(args.train_out, "w")
    val_f   = open(args.val_out, "w")
    lic_f   = open(args.license_manifest, "w", newline="")
    csv.writer(lic_f).writerow(["dataset","approx_records","license_hint"])

    total_train = total_val = 0

    for ds_name in args.datasets:
        ds = stream(ds_name)
        count = 0
        for idx, ex in enumerate(ds):
            text = ex.get("content") or ex.get("code") or ex.get("text") or ""
            if not text: 
                continue
            if len(text.encode("utf-8")) > args.max_bytes:
                continue

            # dedupe (very lightweight)
            mh = minhash(text)
            key = f"{ds_name}-{idx}"
            if lsh.query(mh):
                continue
            lsh.insert(key, mh)

            rec = {"text": text}
            if random.random() < args.val_ratio:
                val_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_val += 1
            else:
                train_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_train += 1

            count += 1
            if count >= args.samples_per_dataset:
                break

        csv.writer(lic_f).writerow([ds_name, count, LICENSE_HINTS.get(ds_name.split(":")[0], "Unknown")])
        print(f"[OK] {ds_name}: {count} kept")

    train_f.close(); val_f.close(); lic_f.close()
    print(f"Done: train {total_train} / val {total_val}")
    print(f"Manifest: {args.license_manifest}")

if __name__ == "__main__":
    sys.exit(main())
