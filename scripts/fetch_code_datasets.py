from __future__ import annotations

import argparse
import json
import random
from typing import Callable, Dict, Iterable, Iterator, List

from datasets import load_dataset

from utils import ensure_dirs, jsonl_writer


def _iter_codeparrot(_: List[str]) -> Iterator[str]:
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for ex in ds:
        text = ex.get("content") or ex.get("code")
        if text:
            yield text


def _iter_stack_dedup(languages: List[str]) -> Iterator[str]:
    configs = languages or ["python"]
    for lang in configs:
        ds = load_dataset("bigcode/stack-dedup", name=lang, split="train", streaming=True)
        for ex in ds:
            text = ex.get("content") or ex.get("code")
            if text:
                yield text


def _iter_code_search_net(languages: List[str]) -> Iterator[str]:
    configs = languages or ["python"]
    for lang in configs:
        ds = load_dataset("code_search_net", name=lang, split="train", streaming=True)
        for ex in ds:
            text = ex.get("code") or ex.get("func_code_string") or ex.get("canonical_solution")
            if text:
                yield text


DATASET_LOADERS: Dict[str, Callable[[List[str]], Iterable[str]]] = {
    "codeparrot": _iter_codeparrot,
    "stack_dedup": _iter_stack_dedup,
    "code_search_net": _iter_code_search_net,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and merge multiple open-source code datasets into CLM JSONL files."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["codeparrot", "stack_dedup", "code_search_net"],
        help="Datasets to stream (use 'all' to include every supported source).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["python"],
        help="Language configs for datasets that support them (stack_dedup/code_search_net).",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=50000,
        help="Maximum samples to pull from each dataset on this run (set -1 for all).",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.01,
        help="Fraction of samples routed to the eval/holdout JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for train/eval splitting.",
    )
    parser.add_argument(
        "--train-output",
        default="data/processed/pretrain-train.jsonl",
        help="Path to the aggregated train JSONL.",
    )
    parser.add_argument(
        "--eval-output",
        default="data/processed/pretrain-eval.jsonl",
        help="Path to the aggregated eval JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    dataset_names = args.datasets
    if "all" in dataset_names:
        dataset_names = list(DATASET_LOADERS)

    rng = random.Random(args.seed)
    limit = None if args.limit_per_dataset == -1 else args.limit_per_dataset

    train_file = jsonl_writer(args.train_output, "w")
    eval_file = jsonl_writer(args.eval_output, "w")
    counts = {"train": 0, "eval": 0}
    per_dataset: Dict[str, int] = {}

    try:
        for name in dataset_names:
            if name not in DATASET_LOADERS:
                raise ValueError(f"Unsupported dataset '{name}'. Choices: {list(DATASET_LOADERS)}")
            produced = 0
            for text in DATASET_LOADERS[name](args.languages):
                if not text or not text.strip():
                    continue
                split = "eval" if rng.random() < args.eval_ratio else "train"
                payload = {"text": text, "source": name}
                (eval_file if split == "eval" else train_file).write(json.dumps(payload) + "\n")
                counts[split] += 1
                produced += 1
                if limit is not None and produced >= limit:
                    break
            per_dataset[name] = produced
            print(f"[fetch_code_datasets] {name}: wrote {produced} samples")
    finally:
        train_file.close()
        eval_file.close()

    print(
        f"[fetch_code_datasets] train={counts['train']} eval={counts['eval']} "
        f"dest=({args.train_output}, {args.eval_output})"
    )
    for name, produced in per_dataset.items():
        print(f"  - {name}: {produced}")


if __name__ == "__main__":
    main()
