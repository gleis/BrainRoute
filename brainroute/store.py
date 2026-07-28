from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from .config_paths import DATA_DIR


SCHEMA = """
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
create index if not exists idx_events_created on events(created_at);
create index if not exists idx_events_type on events(event_type, created_at);
create index if not exists idx_messages_session on messages(session_id, id);
create index if not exists idx_runs_created on runs(created_at);
create index if not exists idx_runs_model on runs(model_id, created_at);
create index if not exists idx_sessions_updated on sessions(updated_at);
"""

# One connection per thread. The previous implementation opened a fresh
# connection and replayed the full schema on every single call, which made each
# telemetry write cost several filesystem round trips.
_local = threading.local()
_init_lock = threading.Lock()
_initialised = False


def write_event(event: dict[str, Any]) -> None:
    with _connect() as db:
        db.execute(
            "insert into events(created_at, event_type, payload) values (?, ?, ?)",
            (_now(), event.get("event", "event"), json.dumps(event, sort_keys=True)),
        )


def read_events(limit: int = 50, event_type: str | None = None) -> list[dict[str, Any]]:
    query = "select created_at, payload from events"
    params: list[Any] = []
    if event_type:
        query += " where event_type = ?"
        params.append(event_type)
    query += " order by id desc limit ?"
    params.append(limit)
    with _connect() as db:
        rows = db.execute(query, params).fetchall()
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


def model_stats(window_hours: int = 168, sample_limit: int = 2000) -> dict[str, dict[str, Any]]:
    """Observed per-model behaviour, used to correct declared model scores.

    Returns latency percentiles, success rate, realised cost per run, and the
    current consecutive-failure streak that the circuit breaker keys off.
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    with _connect() as db:
        rows = db.execute(
            """
            select model_id, ok, latency_ms, estimated_cost_usd, output_tokens, created_at
            from runs where created_at >= ? and model_id is not null
            order by id desc limit ?
            """,
            (since, sample_limit),
        ).fetchall()

    stats: dict[str, dict[str, Any]] = {}
    for model_id, ok, latency_ms, cost, output_tokens, created_at in rows:
        entry = stats.setdefault(
            model_id,
            {
                "runs": 0,
                "successes": 0,
                "latencies": [],
                "cost_total": 0.0,
                "output_tokens_total": 0,
                "consecutive_failures": 0,
                "streak_open": True,
                "last_failure_at": None,
                "last_run_at": created_at,
            },
        )
        entry["runs"] += 1
        entry["cost_total"] += float(cost or 0)
        entry["output_tokens_total"] += int(output_tokens or 0)
        if ok:
            entry["successes"] += 1
            # Rows arrive newest-first, so the first success ends the streak.
            entry["streak_open"] = False
            if latency_ms:
                entry["latencies"].append(int(latency_ms))
        else:
            if entry["streak_open"]:
                entry["consecutive_failures"] += 1
            if entry["last_failure_at"] is None:
                entry["last_failure_at"] = created_at

    for entry in stats.values():
        latencies = sorted(entry.pop("latencies"))
        runs = entry["runs"]
        entry.pop("streak_open", None)
        entry["ok_rate"] = round(entry["successes"] / runs, 4) if runs else 0.0
        entry["p50_latency_ms"] = _percentile(latencies, 0.50)
        entry["p95_latency_ms"] = _percentile(latencies, 0.95)
        entry["avg_cost_usd"] = round(entry["cost_total"] / runs, 8) if runs else 0.0
        entry["cost_total"] = round(entry["cost_total"], 8)
    return stats


def feedback_stats(window_hours: int = 720) -> dict[str, dict[str, Any]]:
    """Aggregate thumbs up/down per model so human judgement can nudge routing."""
    since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    with _connect() as db:
        rows = db.execute(
            "select payload from events where event_type = 'feedback' and created_at >= ?",
            (since,),
        ).fetchall()
    stats: dict[str, dict[str, Any]] = {}
    for (payload,) in rows:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        model_id = event.get("selected_model") or event.get("model")
        rating = str(event.get("rating", "")).lower()
        if not model_id or rating not in {"good", "bad", "up", "down"}:
            continue
        entry = stats.setdefault(model_id, {"good": 0, "bad": 0})
        entry["good" if rating in {"good", "up"} else "bad"] += 1
    for entry in stats.values():
        total = entry["good"] + entry["bad"]
        entry["total"] = total
        # Laplace-smoothed so a single vote cannot swing the score.
        entry["score"] = round((entry["good"] + 1) / (total + 2), 4)
    return stats


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


def prune(max_events: int = 50_000, max_runs: int = 50_000) -> dict[str, int]:
    """Cap unbounded tables so a long-running instance does not grow forever."""
    with _connect() as db:
        deleted_events = db.execute(
            "delete from events where id <= (select coalesce(max(id), 0) - ? from events)",
            (max_events,),
        ).rowcount
        deleted_runs = db.execute(
            "delete from runs where id <= (select coalesce(max(id), 0) - ? from runs)",
            (max_runs,),
        ).rowcount
    return {"events_deleted": max(0, deleted_events), "runs_deleted": max(0, deleted_runs)}


@contextmanager
def _connect():
    db = _connection()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise


def _connection() -> sqlite3.Connection:
    global _initialised
    db = getattr(_local, "db", None)
    if db is not None:
        return db
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA_DIR / "brainroute.sqlite", timeout=30, check_same_thread=False)
    db.execute("pragma busy_timeout=30000")
    _enable_wal(db)
    with _init_lock:
        if not _initialised:
            db.executescript(SCHEMA)
            db.commit()
            _initialised = True
    _local.db = db
    return db


def _enable_wal(db: sqlite3.Connection) -> None:
    """Turn on WAL, falling back if the filesystem cannot support it.

    WAL lets readers proceed during writes, which matters as soon as the server
    handles concurrent requests. It needs a shared-memory file alongside the
    database, which some network shares and container mounts cannot provide -
    there the pragma appears to succeed and the next write fails with a disk
    I/O error, so the mode is verified with a real statement.
    """
    try:
        db.execute("pragma journal_mode=WAL")
        db.execute("pragma synchronous=NORMAL")
        db.execute("create table if not exists _wal_probe(id integer primary key)")
        db.execute("drop table if exists _wal_probe")
        db.commit()
        return
    except sqlite3.OperationalError:
        db.rollback()
    try:
        db.execute("pragma journal_mode=DELETE")
        db.execute("pragma synchronous=FULL")
        db.commit()
    except sqlite3.OperationalError:
        # Let the real failure surface from the first query instead of here,
        # where the message would only be about a pragma.
        db.rollback()


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return int(sorted_values[index])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
