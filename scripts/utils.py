from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import TextIO

DATA_DIRS = ["data/raw", "data/processed", "data/eval", "out"]


def ensure_dirs() -> None:
    """Create the canonical project directories if they do not exist."""
    for p in DATA_DIRS:
        Path(p).mkdir(parents=True, exist_ok=True)


def jsonl_writer(path: str | Path, mode: str = "w") -> TextIO:
    """Open a JSONL file for writing/append while ensuring parent directories."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path.open(mode, encoding="utf-8")


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:  # noqa: D401 - HTMLParser hook
        self._chunks.append(data)

    def get_data(self) -> str:
        return "".join(self._chunks)


def html_to_text(value: str | None) -> str:
    """Convert StackOverflow HTML bodies into normalized plain text."""
    if not value:
        return ""
    parser = _HTMLStripper()
    parser.feed(value)
    parser.close()
    text = unescape(parser.get_data())
    return re.sub(r"\s+", " ", text).strip()
