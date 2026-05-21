from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def call_ollama(model: dict[str, Any], prompt: str, timeout: int = 60) -> str:
    endpoint = str(model.get("endpoint", "http://localhost:11434/api/chat"))
    payload = {
        "model": model.get("model", model.get("id")),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return data.get("message", {}).get("content", "")

