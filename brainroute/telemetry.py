from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import read_events as read_db_events
from .store import write_event as write_db_event


def write_event(event: dict[str, Any], path: str | Path = "logs/telemetry.jsonl") -> None:
    write_db_event(event)
    # Keep the JSONL stream for simple local tailing during development.
    import json
    from datetime import datetime, timezone

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), **event}
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_events(path: str | Path = "logs/telemetry.jsonl", limit: int = 50) -> list[dict[str, Any]]:
    db_events = read_db_events(limit=limit)
    if db_events:
        return db_events
    import json

    source = Path(path)
    if not source.exists():
        return []
    lines = source.read_text(encoding="utf-8").splitlines()[-limit:]
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
