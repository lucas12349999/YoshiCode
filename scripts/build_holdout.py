from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from utils import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a JSONL file to create a hold-out eval split.")
    parser.add_argument("source", help="Input JSONL file.")
    parser.add_argument("--size", type=int, default=1000, help="Number of records to sample.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="data/eval/holdout.jsonl",
        help="Destination JSONL for hold-out samples.",
    )
    parser.add_argument(
        "--remainder-output",
        default=None,
        help="Optional JSONL for the remaining records (train minus holdout).",
    )
    return parser.parse_args()


def load_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def dump_lines(path: str, lines: List[str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    lines = load_lines(args.source)
    if not lines:
        raise ValueError("Source file is empty.")
    rng = random.Random(args.seed)
    if args.size >= len(lines):
        holdout = lines
        remainder = []
    else:
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        hold_indices = set(indices[: args.size])
        holdout = [lines[i] for i in hold_indices]
        remainder = [lines[i] for i in indices[args.size:]]
    dump_lines(args.output, holdout)
    if args.remainder_output:
        dump_lines(args.remainder_output, remainder)
    print(
        f"[build_holdout] wrote {len(holdout)} rows → {args.output}"
        + (f" | remainder → {args.remainder_output}" if args.remainder_output else "")
    )


if __name__ == "__main__":
    main()
