from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def call_openrouter(model: dict[str, Any], prompt: str, timeout: int = 60) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter models")
    request = urllib.request.Request(
        str(model.get("endpoint", "https://openrouter.ai/api/v1/chat/completions")),
        data=json.dumps({
            "model": model.get("model", model.get("id")),
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gleis/BrainRoute",
            "X-Title": "BrainRoute",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {detail}") from exc
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
    choices = data.get("choices", [])
    return choices[0].get("message", {}).get("content", "") if choices else ""
