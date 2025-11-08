#!/usr/bin/env python3
import json, argparse, random
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a=ap.parse_args()
    random.seed(a.seed)
    lines = [l for _,l in zip(range(200000), open(a.train))]
    sample = random.sample(lines, min(a.n, len(lines)))
    with open(a.out,"w") as f:
        for l in sample: f.write(l)
    print(f"[OK] wrote holdout → {a.out}")
if __name__=="__main__":
    main()
