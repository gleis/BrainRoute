"""Observed-behaviour layer: turns telemetry into routing signal.

Declared scores in ``models.yaml`` are a prior. This module supplies the
posterior: what a model actually did recently. It answers two questions the
router needs before it can be trusted in production.

1. Should this model be considered at all right now? (circuit breaker)
2. Are its declared speed/quality numbers still believable? (blending)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .cache import TTLCache
from .settings import load_settings
from .store import feedback_stats, model_stats


CLOSED = "closed"
HALF_OPEN = "half_open"
OPEN = "open"

# Telemetry aggregation is a few SQL queries; recomputing it per candidate model
# per request would dominate routing latency.
_SNAPSHOT_CACHE = TTLCache(ttl_seconds=15.0, max_entries=4)

# Latency at which the derived speed score is exactly 0.5. Chosen so that
# sub-second responses score high and 30s responses score near zero.
LATENCY_HALF_POINT_MS = 4000.0


@dataclass(frozen=True)
class ModelReliability:
    model_id: str
    state: str = CLOSED
    samples: int = 0
    ok_rate: float = 1.0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0
    consecutive_failures: int = 0
    avg_cost_usd: float = 0.0
    feedback_score: float | None = None
    feedback_votes: int = 0
    detail: str = "no telemetry yet"

    @property
    def available(self) -> bool:
        return self.state != OPEN

    @property
    def confidence(self) -> float:
        """How much weight observed data should carry, from 0 to 1."""
        return min(1.0, self.samples / max(1, _config()["min_samples"]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "state": self.state,
            "samples": self.samples,
            "ok_rate": self.ok_rate,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "consecutive_failures": self.consecutive_failures,
            "avg_cost_usd": self.avg_cost_usd,
            "feedback_score": self.feedback_score,
            "feedback_votes": self.feedback_votes,
            "detail": self.detail,
        }


def snapshot(force_refresh: bool = False) -> dict[str, ModelReliability]:
    """Per-model reliability keyed by model id."""
    if force_refresh:
        _SNAPSHOT_CACHE.clear()
    return _SNAPSHOT_CACHE.get_or_set("snapshot", _build_snapshot)


def reliability_for(model_id: str) -> ModelReliability:
    return snapshot().get(model_id) or ModelReliability(model_id=model_id)


def invalidate() -> None:
    _SNAPSHOT_CACHE.clear()


def observed_speed_score(latency_ms: float) -> float:
    """Map a latency in milliseconds onto a 0-1 speed score."""
    if latency_ms <= 0:
        return 0.5
    return round(1.0 / (1.0 + (latency_ms / LATENCY_HALF_POINT_MS)), 4)


def blend(declared: float, observed: float, confidence: float) -> float:
    """Move from the declared prior toward the observation as evidence accrues."""
    weight = max(0.0, min(1.0, confidence)) * _config()["max_observed_weight"]
    return round(declared * (1 - weight) + observed * weight, 4)


def _build_snapshot() -> dict[str, ModelReliability]:
    config = _config()
    if not config["enabled"]:
        return {}
    runs = model_stats(window_hours=config["window_hours"])
    votes = feedback_stats()
    now = datetime.now(timezone.utc)

    result: dict[str, ModelReliability] = {}
    for model_id, entry in runs.items():
        vote = votes.get(model_id)
        state, detail = _circuit_state(entry, config, now)
        result[model_id] = ModelReliability(
            model_id=model_id,
            state=state,
            samples=int(entry["runs"]),
            ok_rate=float(entry["ok_rate"]),
            p50_latency_ms=int(entry["p50_latency_ms"]),
            p95_latency_ms=int(entry["p95_latency_ms"]),
            consecutive_failures=int(entry["consecutive_failures"]),
            avg_cost_usd=float(entry["avg_cost_usd"]),
            feedback_score=float(vote["score"]) if vote else None,
            feedback_votes=int(vote["total"]) if vote else 0,
            detail=detail,
        )

    # Models with feedback but no runs in the window still deserve an entry.
    for model_id, vote in votes.items():
        if model_id in result:
            continue
        result[model_id] = ModelReliability(
            model_id=model_id,
            feedback_score=float(vote["score"]),
            feedback_votes=int(vote["total"]),
            detail="feedback only",
        )
    return result


def _circuit_state(entry: dict[str, Any], config: dict[str, Any], now: datetime) -> tuple[str, str]:
    failures = int(entry["consecutive_failures"])
    if failures < config["failure_threshold"]:
        return CLOSED, f"{entry['runs']} runs, {entry['ok_rate']:.0%} ok"

    last_failure = _parse(entry.get("last_failure_at"))
    if last_failure is None:
        return CLOSED, "failure timestamp unavailable"

    age_seconds = (now - last_failure).total_seconds()
    if age_seconds < config["cooldown_seconds"]:
        remaining = int(config["cooldown_seconds"] - age_seconds)
        return OPEN, f"circuit open after {failures} failures, retry in {remaining}s"
    # Cooldown elapsed: let one request through to probe recovery, but keep the
    # scoring penalty so a healthy peer still wins a close race.
    return HALF_OPEN, f"probing recovery after {failures} failures"


def _config() -> dict[str, Any]:
    settings = load_settings().get("reliability", {})
    return {
        "enabled": bool(settings.get("enabled", True)),
        "window_hours": int(settings.get("window_hours", 168)),
        "failure_threshold": max(1, int(settings.get("failure_threshold", 3))),
        "cooldown_seconds": max(1, int(settings.get("cooldown_seconds", 120))),
        "min_samples": max(1, int(settings.get("min_samples", 8))),
        "max_observed_weight": max(0.0, min(1.0, float(settings.get("max_observed_weight", 0.7)))),
    }


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
