from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from .config_paths import DATA_DIR


def write_event(event: dict[str, Any]) -> None:
    with _connect() as db:
        db.execute(
            "insert into events(created_at, event_type, payload) values (?, ?, ?)",
            (_now(), event.get("event", "event"), json.dumps(event, sort_keys=True)),
        )


def read_events(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as db:
        rows = db.execute(
            "select created_at, payload from events order by id desc limit ?",
            (limit,),
        ).fetchall()
    events = []
    for created_at, payload in reversed(rows):
        try:
            events.append({"created_at": created_at, **json.loads(payload)})
        except json.JSONDecodeError:
            continue
    return events


def create_session(session_id: str | None = None) -> str:
    resolved = session_id or str(uuid.uuid4())
    with _connect() as db:
        db.execute(
            "insert or ignore into sessions(id, created_at, updated_at) values (?, ?, ?)",
            (resolved, _now(), _now()),
        )
    return resolved


def append_message(session_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
    session_id = create_session(session_id)
    with _connect() as db:
        db.execute(
            "insert into messages(session_id, created_at, role, content, meta) values (?, ?, ?, ?, ?)",
            (session_id, _now(), role, content, json.dumps(meta or {}, sort_keys=True)),
        )
        db.execute("update sessions set updated_at = ? where id = ?", (_now(), session_id))


def get_session(session_id: str) -> dict[str, Any]:
    with _connect() as db:
        rows = db.execute(
            "select created_at, role, content, meta from messages where session_id = ? order by id",
            (session_id,),
        ).fetchall()
    return {
        "id": session_id,
        "messages": [
            {
                "created_at": created_at,
                "role": role,
                "content": content,
                "meta": _json(meta),
            }
            for created_at, role, content, meta in rows
        ],
    }


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as db:
        rows = db.execute(
            """
            select sessions.id, sessions.updated_at, count(messages.id)
            from sessions left join messages on sessions.id = messages.session_id
            group by sessions.id order by sessions.updated_at desc limit ?
            """,
            (limit,),
        ).fetchall()
    return [{"id": item[0], "updated_at": item[1], "messages": item[2]} for item in rows]


def record_run(run: dict[str, Any]) -> None:
    with _connect() as db:
        db.execute(
            """
            insert into runs(created_at, model_id, provider, latency_ms, input_tokens, output_tokens, estimated_cost_usd, ok, payload)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                run.get("model_id"),
                run.get("provider"),
                run.get("latency_ms", 0),
                run.get("input_tokens", 0),
                run.get("output_tokens", 0),
                run.get("estimated_cost_usd", 0),
                bool(run.get("ok")),
                json.dumps(run, sort_keys=True),
            ),
        )


def spend_since(iso_prefix: str) -> float:
    with _connect() as db:
        row = db.execute(
            "select coalesce(sum(estimated_cost_usd), 0) from runs where created_at like ?",
            (f"{iso_prefix}%",),
        ).fetchone()
    return float(row[0])


def dashboard() -> dict[str, Any]:
    with _connect() as db:
        rows = db.execute(
            """
            select model_id, count(*), avg(latency_ms), sum(case when ok then 1 else 0 end),
                   sum(estimated_cost_usd)
            from runs group by model_id order by count(*) desc
            """
        ).fetchall()
    return {
        "models": [
            {
                "model_id": row[0],
                "runs": row[1],
                "avg_latency_ms": round(row[2] or 0, 1),
                "ok_rate": round((row[3] or 0) / row[1], 3) if row[1] else 0,
                "estimated_cost_usd": round(row[4] or 0, 6),
            }
            for row in rows
        ]
    }


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA_DIR / "brainroute.sqlite", timeout=10)
    db.executescript(
        """
        create table if not exists events(
            id integer primary key, created_at text not null, event_type text not null, payload text not null
        );
        create table if not exists sessions(
            id text primary key, created_at text not null, updated_at text not null
        );
        create table if not exists messages(
            id integer primary key, session_id text not null references sessions(id),
            created_at text not null, role text not null, content text not null, meta text not null
        );
        create table if not exists runs(
            id integer primary key, created_at text not null, model_id text, provider text,
            latency_ms integer, input_tokens integer, output_tokens integer,
            estimated_cost_usd real, ok integer, payload text not null
        );
        """
    )
    try:
        yield db
        db.commit()
    finally:
        db.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
