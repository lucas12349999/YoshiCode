from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, Tuple

from utils import ensure_dirs, jsonl_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple instruction datasets plus StackOverflow Q&A into a single SFT JSONL."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="JSONL files that already contain instruction/input/output triples.",
    )
    parser.add_argument(
        "--stackoverflow",
        nargs="*",
        default=[],
        help="Raw StackOverflow Q&A JSONL files produced by fetch_stackoverflow.py.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/sft.jsonl",
        help="Destination JSONL file (use '-' for stdout).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the destination file instead of overwriting.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include source metadata (e.g., StackOverflow links) inside each JSON record.",
    )
    return parser.parse_args()


def iter_jsonl(paths: Iterable[str]):
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, idx, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse {path}:{idx}") from exc


def normalize_record(record: dict, include_metadata: bool) -> Tuple[str, str, str, dict | None] | None:
    inst = record.get("instruction") or record.get("prompt")
    outp = record.get("output") or record.get("response")
    inp = record.get("input", "")
    if inst and outp:
        meta = record.get("metadata") if include_metadata else None
        return inst, inp, outp, meta

    title = record.get("question_title")
    answer = record.get("answer_body")
    if title and answer:
        question = record.get("question_body") or ""
        inst = (
            "You are helping with a StackOverflow question.\n"
            f"Title: {title}\n\nQuestion:\n{question}\n\nProvide a concise, helpful answer."
        )
        output = answer
        meta = None
        if include_metadata:
            meta = {
                "source": record.get("source", "stackoverflow"),
                "license": record.get("license", "CC BY-SA 4.0"),
            }
            if record.get("question_link"):
                meta["link"] = record["question_link"]
                output = f"{output}\n\nReference: {record['question_link']} (CC BY-SA 4.0)"
        return inst, "", output, meta
    return None


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out_handle = sys.stdout if args.output == "-" else jsonl_writer(args.output, "a" if args.append else "w")
    written = skipped = 0
    try:
        for source, line_no, record in iter_jsonl([*args.sources, *args.stackoverflow]):
            normalized = normalize_record(record, args.include_metadata)
            if not normalized:
                skipped += 1
                continue
            inst, inp, outp, meta = normalized
            payload = {"instruction": inst, "input": inp, "output": outp}
            if meta:
                payload["metadata"] = meta
            out_handle.write(json.dumps(payload) + "\n")
            written += 1
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()
    print(
        f"[build_sft] wrote {written} records (skipped {skipped}) "
        f"→ {args.output if args.output != '-' else 'stdout'}"
    )


if __name__ == "__main__":
    main()
