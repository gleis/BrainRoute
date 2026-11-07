"""Runtime settings.

Settings are composed in three layers, lowest precedence first:

1. ``DEFAULT_SETTINGS`` - the schema and safe operating point.
2. The active **privacy posture** preset (see :mod:`modelroute.postures`).
3. The user's own saved values.

Only layer 3 is written to disk. That separation is what makes postures useful:
switching posture changes every flag you have not explicitly pinned, while the
ones you did pin survive the switch.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from . import postures
from .config_paths import DATA_DIR


# Context-scoped settings layer used by evals and tests. A ContextVar keeps it
# from leaking across threads in the threaded HTTP server.
_overrides: ContextVar[dict[str, Any] | None] = ContextVar(
    "modelroute_settings_override", default=None
)


DEFAULT_SETTINGS: dict[str, Any] = {
    "models": {},
    "catalog": {
        "ollama_endpoint": "http://127.0.0.1:11434",
        "discovery_ttl_seconds": 60,
        # Ask Ollama for each model's real context window instead of guessing
        # from its name. Costs one /api/show call per model per discovery.
        "detect_local_context": True,
        # Optional OpenAI-compatible gateways to list models from. Each entry:
        #   id            label used for model ids and health output
        #   enabled       off unless you turn it on
        #   models_url    GET endpoint returning {"data": [...]}
        #   chat_url      POST endpoint for /v1/chat/completions
        #   api_key_env   environment variable holding the credential
        #   limit         cap on models imported, so a large catalog cannot
        #                 flood the model list
        "remote_sources": [],
    },
    "classifier": {
        "enabled": False,
        "model_id": "",
        "cache_ttl_seconds": 300,
    },
    "policy": {
        # The posture supplies the three flags below; they are listed here so
        # the shape of the settings document is complete and self-documenting.
        "posture": postures.DEFAULT_POSTURE,
        "prefer_local": True,
        "allow_cloud": True,
        "allow_cloud_for_private": False,
        # Scoring bonus applied to local models when prefer_local is on. Big
        # enough to win a close race, small enough that the capability floor
        # can still escalate a task a local model genuinely cannot handle.
        "local_bonus": 0.15,
        "max_estimated_cost_usd": 0.25,
        "monthly_budget_usd": 50.0,
        "context_headroom_tokens": 512,
    },
    "reliability": {
        "enabled": True,
        "window_hours": 168,
        "failure_threshold": 3,
        "cooldown_seconds": 120,
        "min_samples": 8,
        "max_observed_weight": 0.7,
    },
    "routing": {
        # Minimum credible quality per complexity tier. Lower the "high" floor
        # to let a smaller local model stay in contention on hard tasks, at the
        # cost of answer quality.
        "capability_floor": {"low": 0.0, "medium": 0.55, "high": 0.80},
        "capability_floor_penalty": 1.5,
        # Warn when the capability floor is what pushed a request off local
        # hardware, so a privacy trade is never silent.
        "warn_on_privacy_escalation": True,
        # Epsilon-greedy exploration over near-tied candidates. Off by default
        # so routing stays deterministic and reproducible in evals.
        "exploration_rate": 0.0,
        "exploration_score_margin": 0.05,
        "max_fallback_depth": 2,
    },
    "gateway": {
        "protect_admin_api": True,
        "rate_limit_per_minute": 120,
    },
}


_lock = threading.Lock()
_version = 0


def settings_path() -> Path:
    return DATA_DIR / "settings.json"


def settings_version() -> int:
    """Monotonic counter bumped on every write, used as a cache key."""
    return _version


@contextmanager
def override(update: dict[str, Any]) -> Iterator[None]:
    """Temporarily layer settings on top of the stored ones.

    Scoped to the current context rather than written to disk, so evals can pin
    a posture per case and tests can vary policy without mutating the user's
    configuration or racing other threads.
    """
    global _version
    token = _overrides.set(update)
    _version += 1
    _notify_change()
    try:
        yield
    finally:
        _overrides.reset(token)
        _version += 1
        _notify_change()


def load_settings() -> dict[str, Any]:
    """Defaults, then the posture preset, then stored values, then overrides."""
    stored = load_stored_settings()
    scoped = _overrides.get() or {}
    posture = postures.resolve(
        scoped.get("policy", {}).get("posture") or stored.get("policy", {}).get("posture")
    )

    settings = deepcopy(DEFAULT_SETTINGS)
    _deep_update(settings, postures.preset(posture))
    _deep_update(settings, stored)
    if scoped:
        _deep_update(settings, scoped)

    settings["policy"]["posture"] = posture
    # Derived, read-only facts about the active posture. Handlers consult these
    # rather than re-deriving posture behaviour at each call site.
    settings["policy"]["cloud_override_allowed"] = postures.cloud_override_allowed(posture)
    settings["policy"]["posture_default_profile"] = postures.default_profile(posture)
    return settings


def load_stored_settings() -> dict[str, Any]:
    """Only what the user actually saved - no defaults, no posture."""
    source = settings_path()
    if not source.exists():
        return {}
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt or unreadable settings file must not take the router down;
        # defaults are always a safe operating point.
        return {}
    return stored if isinstance(stored, dict) else {}


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    """Merge into the stored layer and return the fully resolved settings."""
    global _version
    with _lock:
        stored = load_stored_settings()
        _deep_update(stored, _strip_derived(update))
        _atomic_write(settings_path(), json.dumps(stored, indent=2, sort_keys=True))
        _version += 1
    _notify_change()
    return load_settings()


def set_posture(name: str) -> dict[str, Any]:
    """Switch posture, leaving any explicitly pinned flags untouched."""
    return save_settings({"policy": {"posture": postures.resolve(name)}})


def clear_policy_overrides() -> dict[str, Any]:
    """Drop pinned policy flags so the posture fully governs them again."""
    global _version
    with _lock:
        stored = load_stored_settings()
        policy = stored.get("policy", {})
        for key in ("prefer_local", "allow_cloud", "allow_cloud_for_private"):
            policy.pop(key, None)
        if policy:
            stored["policy"] = policy
        else:
            stored.pop("policy", None)
        _atomic_write(settings_path(), json.dumps(stored, indent=2, sort_keys=True))
        _version += 1
    _notify_change()
    return load_settings()


def update_model(model_id: str, update: dict[str, Any]) -> dict[str, Any]:
    stored = load_stored_settings()
    models = stored.get("models", {})
    models.setdefault(model_id, {})
    _deep_update(models[model_id], update)
    return save_settings({"models": models})


def _strip_derived(update: dict[str, Any]) -> dict[str, Any]:
    """Never persist values the loader computes.

    The UI round-trips the whole resolved policy object; without this, derived
    fields would be written back as if the user had pinned them.
    """
    cleaned = deepcopy(update)
    policy = cleaned.get("policy")
    if isinstance(policy, dict):
        policy.pop("cloud_override_allowed", None)
        policy.pop("posture_default_profile", None)
    return cleaned


def _notify_change() -> None:
    # Imported lazily to avoid an import cycle: config and reliability both
    # depend on settings.
    from . import config as config_module
    from . import reliability

    config_module.invalidate()
    reliability.invalidate()


def _atomic_write(path: Path, content: str) -> None:
    """Write via a temp file + rename so a crash cannot truncate settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".settings-", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
