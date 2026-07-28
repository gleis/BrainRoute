"""Telemetry facade.

SQLite is the system of record. The JSONL file is a convenience for local
tailing (``tail -f logs/telemetry.jsonl``) and is size-capped so a long-running
instance cannot fill the disk with it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import read_events as read_db_events
from .store import write_event as write_db_event


DEFAULT_LOG_PATH = "logs/telemetry.jsonl"
MAX_LOG_BYTES = 16 * 1024 * 1024


def write_event(event: dict[str, Any], path: str | Path = DEFAULT_LOG_PATH) -> None:
    write_db_event(event)
    _append_line(Path(path), {"created_at": _now(), **event})


def read_events(
    path: str | Path = DEFAULT_LOG_PATH,
    limit: int = 50,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    db_events = read_db_events(limit=limit, event_type=event_type)
    if db_events:
        return db_events
    return _read_lines(Path(path), limit)


def _append_line(output: Path, payload: dict[str, Any]) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        _rotate(output)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except OSError:
        # The development log is best-effort; losing it must never fail a request.
        pass


def _rotate(output: Path) -> None:
    try:
        if output.exists() and output.stat().st_size > MAX_LOG_BYTES:
            os.replace(output, output.with_suffix(output.suffix + ".1"))
    except OSError:
        pass


def _read_lines(source: Path, limit: int) -> list[dict[str, Any]]:
    if not source.exists():
        return []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
