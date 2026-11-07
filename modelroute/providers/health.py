from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .http import ProviderError, get_json


def provider_health(models: list[dict[str, Any]], timeout: int = 2) -> list[dict[str, Any]]:
    """Check every enabled model's provider, in parallel.

    Sequential checks made the health endpoint as slow as the sum of all
    provider timeouts, which is the worst case exactly when things are broken.
    """
    candidates = [model for model in models if model.get("enabled", True)]
    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        results = list(pool.map(lambda model: _check_model(model, timeout), candidates))
    return _annotate(results)


def _annotate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach observed reliability so health reflects behaviour, not just reachability."""
    from ..reliability import snapshot

    health = snapshot()
    for item in results:
        entry = health.get(item["id"])
        if entry is None:
            continue
        item["circuit"] = entry.state
        item["ok_rate"] = entry.ok_rate
        item["samples"] = entry.samples
        item["p95_latency_ms"] = entry.p95_latency_ms
        if entry.state != "closed":
            item["ok"] = False
            item["detail"] = f"{item['detail']} | {entry.detail}"
    return results


def _check_model(model: dict[str, Any], timeout: int) -> dict[str, Any]:
    provider = model.get("provider")
    if provider == "ollama":
        return _check_ollama(model, timeout)
    if provider in _KEYED_PROVIDERS:
        return _check_api_key(model, provider, _KEYED_PROVIDERS[provider])
    if provider == "openai_compatible":
        return _check_openai_compatible(model)
    return {
        "id": model.get("id"),
        "provider": provider,
        "ok": False,
        "detail": "Provider adapter not implemented",
    }


_KEYED_PROVIDERS = {"openai": "OPENAI_API_KEY"}


def _check_api_key(model: dict[str, Any], provider: str, variable: str) -> dict[str, Any]:
    present = bool(os.environ.get(variable))
    return {
        "id": model.get("id"),
        "provider": provider,
        "ok": present,
        "detail": f"{variable} {'set' if present else 'missing'}",
    }


def _check_openai_compatible(model: dict[str, Any]) -> dict[str, Any]:
    """A gateway is reachable-in-principle if it has an endpoint and a credential."""
    label = str(model.get("provider_label") or "openai_compatible")
    if not model.get("endpoint"):
        return {"id": model.get("id"), "provider": label, "ok": False,
                "detail": "no endpoint configured"}
    variable = str(model.get("api_key_env") or "OPENAI_COMPATIBLE_API_KEY")
    if not model.get("requires_api_key", True):
        return {"id": model.get("id"), "provider": label, "ok": True,
                "detail": "endpoint set, no credential required"}
    present = bool(os.environ.get(variable))
    return {
        "id": model.get("id"),
        "provider": label,
        "ok": present,
        "detail": f"{variable} {'set' if present else 'missing'}",
    }


def _check_ollama(model: dict[str, Any], timeout: int) -> dict[str, Any]:
    endpoint = str(model.get("endpoint", "http://localhost:11434/api/chat"))
    tags_url = endpoint.replace("/api/chat", "/api/tags")
    try:
        data = get_json(tags_url, provider="ollama", timeout=timeout)
    except ProviderError as exc:
        return {
            "id": model.get("id"),
            "provider": model.get("provider"),
            "ok": False,
            "detail": f"Ollama unavailable: {exc}",
        }
    wanted = model.get("model")
    names = {
        name
        for item in data.get("models", [])
        for name in (item.get("name"), item.get("model"))
        if name
    }
    available = ", ".join(sorted(names)) or "no Ollama models reported"
    return {
        "id": model.get("id"),
        "provider": model.get("provider"),
        "ok": wanted in names,
        "detail": (
            f"Ollama has {wanted}"
            if wanted in names
            else f"Ollama reachable; missing {wanted}. Available: {available}"
        ),
    }
