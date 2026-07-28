from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import postures
from .cache import TTLCache
from .pricing import apply_pricing
from .settings import load_settings, settings_version


# Discovery hits the network. Without a cache, every /api/config, /api/route and
# /v1/chat/completions call would block on live HTTP to every configured source.
_DISCOVERY_CACHE = TTLCache(ttl_seconds=60.0, max_entries=8)


def discover_models(timeout: int = 4, force_refresh: bool = False) -> dict[str, Any]:
    """Discover runtime-available models, cached for the configured TTL."""
    settings = load_settings()
    ttl = float(settings["catalog"].get("discovery_ttl_seconds", 60) or 0)
    if force_refresh or ttl <= 0:
        result = _discover_uncached(settings, timeout)
        if ttl > 0:
            _DISCOVERY_CACHE.ttl_seconds = ttl
            _DISCOVERY_CACHE.set(_cache_key(timeout), result)
        return result
    _DISCOVERY_CACHE.ttl_seconds = ttl
    return _DISCOVERY_CACHE.get_or_set(
        _cache_key(timeout), lambda: _discover_uncached(settings, timeout)
    )


def invalidate() -> None:
    _DISCOVERY_CACHE.clear()


def _cache_key(timeout: int) -> tuple[Any, ...]:
    return ("discovery", timeout, settings_version())


def _discover_uncached(settings: dict[str, Any], timeout: int) -> dict[str, Any]:
    results: dict[str, Any] = {"models": [], "sources": []}
    ollama_endpoint = settings["catalog"]["ollama_endpoint"]
    try:
        local = discover_ollama(ollama_endpoint, timeout=timeout)
        results["models"].extend(local)
        results["sources"].append({"id": "ollama", "ok": True, "count": len(local)})
    except RuntimeError as exc:
        results["sources"].append({"id": "ollama", "ok": False, "detail": str(exc)})

    posture = settings["policy"].get("posture")
    for source in settings["catalog"].get("remote_sources") or []:
        if not source.get("enabled"):
            continue
        name = str(source.get("id") or "remote")
        if not postures.remote_discovery_allowed(posture):
            # A hard posture constraint, not a preset. A preset alone would be
            # overridden by a pinned setting, quietly breaking the air-gapped
            # guarantee.
            results["sources"].append({
                "id": name,
                "ok": False,
                "detail": f"remote catalog discovery is blocked by the '{posture}' posture",
            })
            continue
        try:
            models = discover_remote(source, timeout=timeout)
            results["models"].extend(models)
            results["sources"].append({"id": name, "ok": True, "count": len(models)})
        except RuntimeError as exc:
            results["sources"].append({"id": name, "ok": False, "detail": str(exc)})
    return results


def discover_ollama(base_url: str, timeout: int = 4) -> list[dict[str, Any]]:
    data = _read_json(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    models = [_ollama_model(item, base_url) for item in data.get("models", [])]
    if models and load_settings()["catalog"].get("detect_local_context", True):
        _attach_context_windows(models, base_url, timeout)
    return models


def _attach_context_windows(
    models: list[dict[str, Any]], base_url: str, timeout: int
) -> None:
    """Replace guessed context windows with what each model actually reports.

    ``/api/tags`` does not include context length, so it takes one ``/api/show``
    call per model. They run in parallel and are best-effort: a failure leaves
    the heuristic guess in place rather than failing discovery. Results are
    cached with the rest of the catalog, so this is not a per-request cost.
    """
    with ThreadPoolExecutor(max_workers=min(8, len(models))) as pool:
        detected = pool.map(
            lambda model: _show_context_length(base_url, model["model"], timeout), models
        )
    for model, length in zip(models, detected):
        if length:
            model["max_context"] = length
            model["context_source"] = "detected"


def _show_context_length(base_url: str, name: str, timeout: int) -> int | None:
    """Read ``<architecture>.context_length`` from Ollama's model metadata."""
    try:
        data = _post_json(f"{base_url.rstrip('/')}/api/show", {"model": name}, timeout=timeout)
    except RuntimeError:
        return None
    info = data.get("model_info") or {}
    architecture = str(info.get("general.architecture", ""))
    candidates = [f"{architecture}.context_length"] if architecture else []
    # Fall back to any architecture-prefixed context_length key.
    candidates.extend(key for key in info if key.endswith(".context_length"))
    for key in candidates:
        value = _positive_int(info.get(key))
        if value:
            return value
    return None


def discover_remote(source: dict[str, Any], timeout: int = 8) -> list[dict[str, Any]]:
    """List models from any endpoint that serves an OpenAI-style ``/models``.

    The response shape is the OpenAI convention - ``{"data": [...]}`` - which
    most gateways follow. Pricing and context length are read when present and
    left unset otherwise, so an unpriced model scores mid-market rather than
    looking free.
    """
    url = str(source.get("models_url") or "").strip()
    if not url:
        raise RuntimeError(f"catalog source '{source.get('id')}' has no models_url")
    data = _read_json(url, timeout=timeout, headers=_source_headers(source))
    entries = data.get("data") if isinstance(data.get("data"), list) else data.get("models")
    limit = _positive_int(source.get("limit")) or 0
    items = list(entries or [])
    if limit:
        items = items[:limit]
    return [_remote_model(item, source) for item in items if isinstance(item, dict)]


def _source_headers(source: dict[str, Any]) -> dict[str, str]:
    variable = str(source.get("api_key_env") or "").strip()
    api_key = os.environ.get(variable) if variable else None
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def overlay_models(configured: list[dict[str, Any]], discovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge seed config, runtime discovery, and user overrides into one list.

    Precedence, lowest to highest: discovered model -> configured model ->
    per-model user settings. Pricing is normalised last so that any override of
    a price is reflected in the derived cost score.
    """
    settings = load_settings()
    models = {model["id"]: dict(model) for model in configured}
    for model in discovered:
        models.setdefault(model["id"], model)
    for model_id, update in settings.get("models", {}).items():
        if model_id in models:
            models[model_id].update(update)
    return [apply_pricing(model) for model in models.values()]


def _ollama_model(item: dict[str, Any], base_url: str) -> dict[str, Any]:
    name = item.get("name") or item.get("model") or ""
    details = item.get("details", {})
    remote = bool(item.get("remote_host"))
    parameters = _parameter_billions(details.get("parameter_size", ""), name)
    return {
        "id": _id("ollama", name),
        "name": f"{name} via Ollama",
        "provider": "ollama",
        "model": name,
        "endpoint": f"{base_url.rstrip('/')}/api/chat",
        "local": not remote,
        "enabled": False,
        "discovered": True,
        "strengths": _local_strengths(details.get("family", ""), name),
        "weaknesses": _local_weaknesses(parameters),
        "privacy_score": 0.4 if remote else 1.0,
        # Capability tracks parameter count; throughput moves the other way.
        "speed_score": _size_speed(parameters),
        "quality_score": _size_quality(parameters),
        "cost_score": 1.0 if not remote else 0.5,
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
        "max_context": _context_guess(name),
        "context_source": "estimated",
        "parameter_size": details.get("parameter_size", ""),
        "parameter_billions": parameters,
        "quantization": details.get("quantization_level", ""),
    }


def _remote_model(item: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    source_id = str(source.get("id") or "remote")
    model_name = str(item.get("id") or item.get("name") or "")
    pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
    # Gateways quote per-token; the router works in dollars per million.
    prompt_price = _float(pricing.get("prompt") or pricing.get("input")) * 1_000_000
    completion_price = _float(pricing.get("completion") or pricing.get("output")) * 1_000_000

    model: dict[str, Any] = {
        "id": _id(source_id, model_name),
        "name": item.get("name") or model_name,
        "provider": "openai_compatible",
        "provider_label": source_id,
        "model": model_name,
        "endpoint": source.get("chat_url"),
        "api_key_env": source.get("api_key_env"),
        "local": bool(source.get("local")),
        "enabled": False,
        "discovered": True,
        "strengths": list(source.get("strengths") or ["general"]),
        "privacy_score": _float(source.get("privacy_score")) or 0.35,
        "speed_score": _float(source.get("speed_score")) or 0.6,
        "max_context": _positive_int(item.get("context_length") or item.get("max_context")) or 8192,
    }
    if prompt_price or completion_price:
        model["input_price_per_mtok"] = prompt_price
        model["output_price_per_mtok"] = completion_price
        # Price is the only quality proxy a bare catalog offers; frontier models
        # are consistently priced above commodity ones.
        model["quality_score"] = _price_quality(prompt_price, completion_price)
    else:
        # No price signal means no basis for a quality claim either.
        model["quality_score"] = _float(source.get("quality_score")) or 0.6
    return model


def _read_json(url: str, timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc


def _id(provider: str, model: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in model).strip("-")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return f"{provider}-{clean}"


def _local_strengths(family: str, name: str) -> list[str]:
    haystack = f"{family} {name}".lower()
    strengths = ["general", "private"]
    if any(token in haystack for token in ("qwen", "llama", "mistral", "gemma", "phi")):
        strengths.append("summarization")
    if any(token in haystack for token in ("coder", "code", "qwen", "deepseek", "starcoder")):
        strengths.append("coding")
    if any(token in haystack for token in ("r1", "reason", "think", "qwq")):
        strengths.append("reasoning")
    return sorted(set(strengths))


def _local_weaknesses(parameters: float) -> list[str]:
    # Small local models are not credible for regulated-domain output.
    return ["legal", "medical"] if parameters < 30 else []


def _parameter_billions(parameter_size: str, name: str) -> float:
    for candidate in (parameter_size, name):
        match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", str(candidate))
        if not match:
            continue
        value = float(match.group(1))
        return value / 1000 if match.group(2).lower() == "m" else value
    return 7.0


def _size_quality(parameters: float) -> float:
    if parameters <= 0:
        return 0.5
    return round(max(0.25, min(0.85, 0.32 + 0.10 * math.log2(parameters))), 3)


def _size_speed(parameters: float) -> float:
    if parameters <= 0:
        return 0.6
    return round(max(0.20, min(0.95, 0.95 - 0.07 * math.log2(parameters))), 3)


def _price_quality(prompt_price: float, completion_price: float) -> float:
    blended = 0.75 * prompt_price + 0.25 * completion_price
    if blended <= 0:
        return 0.5
    return round(max(0.35, min(0.95, 0.35 + 0.16 * math.log10(blended / 0.05))), 3)


# Native context windows for common local families, used when /api/show is
# unavailable or does not report one. Ordered longest-token-match first so
# "llama3.3" is not shadowed by "llama3".
FAMILY_CONTEXT = {
    "llama3.3": 131072,
    "llama3.2": 131072,
    "llama3.1": 131072,
    "llama3": 8192,
    "qwen3": 40960,
    "qwen2.5": 32768,
    "qwen2": 32768,
    "deepseek-r1": 65536,
    "deepseek": 65536,
    "mistral-nemo": 131072,
    "mistral": 32768,
    "mixtral": 32768,
    "phi4": 16384,
    "phi3": 131072,
    "gemma3": 131072,
    "gemma2": 8192,
    "gemma": 8192,
    "command-r": 131072,
    "codellama": 16384,
}

DEFAULT_LOCAL_CONTEXT = 8192


def _context_guess(name: str) -> int:
    """Best guess before /api/show is consulted.

    An explicit size in the tag wins, then a known family, then a conservative
    default. Guessing too low is the safer error: it escalates a request rather
    than sending a prompt that gets silently truncated.
    """
    lowered = str(name).lower()
    match = re.search(r"(\d+)k", lowered)
    if match:
        return int(match.group(1)) * 1024
    for family in sorted(FAMILY_CONTEXT, key=len, reverse=True):
        if lowered.startswith(family) or f"/{family}" in lowered:
            return FAMILY_CONTEXT[family]
    return DEFAULT_LOCAL_CONTEXT


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
