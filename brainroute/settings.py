from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config_paths import DATA_DIR


DEFAULT_SETTINGS: dict[str, Any] = {
    "models": {},
    "catalog": {
        "ollama_endpoint": "http://127.0.0.1:11434",
        "openrouter_enabled": False,
    },
    "classifier": {
        "enabled": False,
        "model_id": "",
    },
    "policy": {
        "prefer_local": False,
        "allow_cloud": True,
        "allow_cloud_for_private": False,
        "max_estimated_cost_usd": 0.25,
        "monthly_budget_usd": 50.0,
    },
}


def settings_path() -> Path:
    return DATA_DIR / "settings.json"


def load_settings() -> dict[str, Any]:
    settings = deepcopy(DEFAULT_SETTINGS)
    source = settings_path()
    if source.exists():
        try:
            _deep_update(settings, json.loads(source.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return settings


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    _deep_update(settings, update)
    source = settings_path()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    return settings


def update_model(model_id: str, update: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    settings["models"].setdefault(model_id, {})
    _deep_update(settings["models"][model_id], update)
    return save_settings({"models": settings["models"]})


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
