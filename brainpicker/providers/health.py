from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def provider_health(models: list[dict[str, Any]], timeout: int = 2) -> list[dict[str, Any]]:
    return [_check_model(model, timeout=timeout) for model in models if model.get("enabled", True)]


def _check_model(model: dict[str, Any], timeout: int) -> dict[str, Any]:
    provider = model.get("provider")
    if provider == "ollama":
        return _check_ollama(model, timeout)
    if provider == "openai":
        return {
            "id": model.get("id"),
            "provider": provider,
            "ok": bool(os.environ.get("OPENAI_API_KEY")),
            "detail": "OPENAI_API_KEY set" if os.environ.get("OPENAI_API_KEY") else "OPENAI_API_KEY missing",
        }
    return {
        "id": model.get("id"),
        "provider": provider,
        "ok": False,
        "detail": "Provider adapter not implemented",
    }


def _check_ollama(model: dict[str, Any], timeout: int) -> dict[str, Any]:
    endpoint = str(model.get("endpoint", "http://localhost:11434/api/chat"))
    tags_url = endpoint.replace("/api/chat", "/api/tags")
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "id": model.get("id"),
            "provider": model.get("provider"),
            "ok": False,
            "detail": str(exc),
        }
    wanted = model.get("model")
    names = {item.get("name") for item in data.get("models", [])}
    return {
        "id": model.get("id"),
        "provider": model.get("provider"),
        "ok": wanted in names,
        "detail": f"{wanted} available" if wanted in names else f"{wanted} not found in Ollama",
    }

