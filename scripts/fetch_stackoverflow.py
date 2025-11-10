from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterator, Optional

import requests


def safe_get(
    url: str,
    params: Optional[dict] = None,
    retries: int = 5,
    delay: float = 1.0,
    timeout: float = 30.0,
):
    """Retry StackExchange API requests with exponential backoff on transient errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {400, 429, 500, 502, 503, 504} or attempt == retries:
                raise
            print(f"[safe_get] HTTP {status} for {url} (attempt {attempt}/{retries}), backing off...")
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            print(f"[safe_get] Request error {exc} (attempt {attempt}/{retries}), backing off...")
        time.sleep(delay * attempt)
    raise RuntimeError(f"Failed after {retries} retries: {url}")




from utils import ensure_dirs, html_to_text, jsonl_writer

API_ROOT = "https://api.stackexchange.com/2.3"
LICENSE = "CC BY-SA 4.0"


def fetch_answer(answer_id: int, key: Optional[str]) -> Optional[Dict]:
    params = {"order": "desc", "sort": "activity", "site": "stackoverflow", "filter": "withbody"}
    if key:
        params["key"] = key
    resp = safe_get(f"{API_ROOT}/answers/{answer_id}", params=params)
    data = resp.json()
    answers = data.get("items", [])
    return answers[0] if answers else None


def fetch_questions(tag: str, pages: int, page_size: int, min_score: int, key: Optional[str]) -> Iterator[Dict]:
    for page in range(1, pages + 1):
        params = {
            "page": page,
            "pagesize": page_size,
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "tagged": tag,
            "filter": "withbody",
        }
        if key:
            params["key"] = key
        resp = safe_get(f"{API_ROOT}/questions", params=params)
        payload = resp.json()
        items = payload.get("items", [])
        backoff = payload.get("backoff")
        if backoff:
            print(f"[fetch_stackoverflow] API requested backoff={backoff}s")
        for item in items:
            if item.get("score", 0) < min_score:
                continue
            ans_id = item.get("accepted_answer_id")
            if not ans_id:
                continue
            answer = fetch_answer(ans_id, key)
            if not answer:
                continue
            yield {
                "question_id": item["question_id"],
                "question_title": item.get("title"),
                "question_body": html_to_text(item.get("body")),
                "question_tags": item.get("tags", []),
                "question_link": item.get("link"),
                "answer_id": answer.get("answer_id"),
                "answer_body": html_to_text(answer.get("body")),
                "score": item.get("score"),
                "source": "stackoverflow",
                "license": LICENSE,
            }
        if not payload.get("has_more"):
            break
        time.sleep(backoff or 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch StackOverflow Q&A pairs (with accepted answers) for SFT datasets."
    )
    parser.add_argument("--tag", default="python", help="StackOverflow tag to query.")
    parser.add_argument("--pages", type=int, default=10, help="Number of result pages to pull.")
    parser.add_argument("--page-size", type=int, default=50, help="Questions per page (max 100).")
    parser.add_argument("--min-score", type=int, default=5, help="Minimum question score to keep.")
    parser.add_argument(
        "--output",
        default="data/raw/stackoverflow_python.jsonl",
        help="Destination JSONL for raw Q&A pairs.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("STACKEXCHANGE_KEY"),
        help="Optional StackExchange API key (env STACKEXCHANGE_KEY).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out_file = jsonl_writer(args.output, "w")
    count = 0
    try:
        for record in fetch_questions(args.tag, args.pages, args.page_size, args.min_score, args.api_key):
            out_file.write(json.dumps(record) + "\n")
            count += 1
    finally:
        out_file.close()
    print(f"[fetch_stackoverflow] wrote {count} records → {args.output} (tag={args.tag})")


if __name__ == "__main__":
    main()
