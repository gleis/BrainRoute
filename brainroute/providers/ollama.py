from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def call_ollama(model: dict[str, Any], prompt: str, timeout: int = 60) -> str:
    return _chat(model, prompt, timeout=timeout).get("message", {}).get("content", "")


def call_ollama_json(
    model: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    timeout: int = 20,
) -> dict[str, Any]:
    content = _chat(
        model,
        prompt,
        timeout=timeout,
        extra={
            "format": schema,
            "options": {"temperature": 0},
        },
    ).get("message", {}).get("content", "")
    return json.loads(content)


def _chat(
    model: dict[str, Any],
    prompt: str,
    timeout: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    endpoint = str(model.get("endpoint", "http://localhost:11434/api/chat"))
    payload = {
        "model": model.get("model", model.get("id")),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    payload.update(extra or {})
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return data
