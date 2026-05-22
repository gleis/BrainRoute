from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .settings import load_settings


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def discover_models(timeout: int = 4) -> dict[str, Any]:
    settings = load_settings()
    results = {"models": [], "sources": []}
    ollama_endpoint = settings["catalog"]["ollama_endpoint"]
    try:
        local = discover_ollama(ollama_endpoint, timeout=timeout)
        results["models"].extend(local)
        results["sources"].append({"id": "ollama", "ok": True, "count": len(local)})
    except RuntimeError as exc:
        results["sources"].append({"id": "ollama", "ok": False, "detail": str(exc)})

    if settings["catalog"].get("openrouter_enabled"):
        try:
            cloud = discover_openrouter(timeout=timeout)
            results["models"].extend(cloud)
            results["sources"].append({"id": "openrouter", "ok": True, "count": len(cloud)})
        except RuntimeError as exc:
            results["sources"].append({"id": "openrouter", "ok": False, "detail": str(exc)})
    return results


def discover_ollama(base_url: str, timeout: int = 4) -> list[dict[str, Any]]:
    data = _read_json(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    return [_ollama_model(item, base_url) for item in data.get("models", [])]


def discover_openrouter(timeout: int = 8) -> list[dict[str, Any]]:
    data = _read_json(OPENROUTER_MODELS_URL, timeout=timeout)
    return [_openrouter_model(item) for item in data.get("data", [])]


def overlay_models(configured: list[dict[str, Any]], discovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = load_settings()
    models = {model["id"]: dict(model) for model in configured}
    for model in discovered:
        models.setdefault(model["id"], model)
    for model_id, update in settings.get("models", {}).items():
        if model_id in models:
            models[model_id].update(update)
    return list(models.values())


def _ollama_model(item: dict[str, Any], base_url: str) -> dict[str, Any]:
    name = item.get("name") or item.get("model")
    details = item.get("details", {})
    remote = bool(item.get("remote_host"))
    return {
        "id": _id("ollama", name),
        "name": f"{name} via Ollama",
        "provider": "ollama",
        "model": name,
        "endpoint": f"{base_url.rstrip('/')}/api/chat",
        "local": not remote,
        "enabled": False,
        "discovered": True,
        "strengths": _local_strengths(details.get("family", "")),
        "privacy_score": 0.4 if remote else 1.0,
        "speed_score": 0.65,
        "quality_score": 0.6,
        "cost_score": 1.0 if not remote else 0.5,
        "max_context": 8192,
        "parameter_size": details.get("parameter_size", ""),
        "quantization": details.get("quantization_level", ""),
    }


def _openrouter_model(item: dict[str, Any]) -> dict[str, Any]:
    pricing = item.get("pricing", {})
    prompt_price = _float(pricing.get("prompt"))
    completion_price = _float(pricing.get("completion"))
    return {
        "id": _id("openrouter", item.get("id", "")),
        "name": item.get("name") or item.get("id"),
        "provider": "openrouter",
        "model": item.get("id"),
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "local": False,
        "enabled": False,
        "discovered": True,
        "strengths": ["general"],
        "privacy_score": 0.35,
        "speed_score": 0.6,
        "quality_score": 0.7,
        "cost_score": _cost_score(prompt_price, completion_price),
        "input_cost_per_token": prompt_price,
        "output_cost_per_token": completion_price,
        "max_context": item.get("context_length") or 8192,
    }


def _read_json(url: str, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc


def _id(provider: str, model: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in model).strip("-")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return f"{provider}-{clean}"


def _local_strengths(family: str) -> list[str]:
    return ["general", "summarization", "private", "coding"] if "qwen" in family else ["general", "private"]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cost_score(prompt_price: float, completion_price: float) -> float:
    total = prompt_price + completion_price
    if total <= 0:
        return 1.0
    return round(max(0.05, min(0.95, 1 - total * 100000)), 3)

