from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_text(path: str | Path, content: str) -> None:
    """Replace a text file atomically so an interrupted write stays recoverable."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    # Sync clients and virus scanners can hold a destination file briefly on
    # Windows. Retry the atomic rename instead of losing a multi-hour run.
    for attempt in range(30):
        try:
            os.replace(temporary, output)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(0.25 * (attempt + 1), 1.0))


def atomic_write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    atomic_write_text(path, content)
