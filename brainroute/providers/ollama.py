from __future__ import annotations

import json
from typing import Any, Iterator

from .http import ProviderError, post_json, stream_lines


PROVIDER = "ollama"
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"

# Ollama defaults num_ctx to roughly 4k regardless of what the model supports,
# and silently truncates anything longer. The router reasons about the model's
# full window, so the request has to ask for the window it was promised.
OLLAMA_DEFAULT_NUM_CTX = 4096

# Context is requested in buckets so that memory use scales with the prompt
# rather than jumping straight to a model's ceiling. A 128k-capable model
# serving a 2k prompt should not allocate a 128k KV cache.
NUM_CTX_BUCKETS = (4096, 8192, 16384, 32768, 65536, 131072)


def call_ollama(
    model: dict[str, Any],
    messages: list[dict[str, str]],
    timeout: int = 60,
    context_tokens: int = 0,
) -> str:
    data = post_json(
        _endpoint(model),
        _payload(model, messages, stream=False, context_tokens=context_tokens),
        provider=PROVIDER,
        timeout=timeout,
    )
    return data.get("message", {}).get("content", "")


def resolve_num_ctx(model: dict[str, Any], context_tokens: int) -> int | None:
    """Smallest bucket that holds the request, capped by the model's window.

    Returns None when the default is already sufficient, so short prompts do
    not carry a num_ctx option at all.
    """
    if context_tokens <= 0:
        return None
    max_context = int(model.get("max_context", 0) or 0)
    if context_tokens <= OLLAMA_DEFAULT_NUM_CTX:
        return None
    for bucket in NUM_CTX_BUCKETS:
        if bucket >= context_tokens:
            return min(bucket, max_context) if max_context else bucket
    return min(context_tokens, max_context) if max_context else context_tokens


def call_ollama_json(
    model: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    timeout: int = 20,
) -> dict[str, Any]:
    """Constrained-decoding call used by the structured classifier."""
    payload = _payload(
        model,
        [{"role": "user", "content": prompt}],
        stream=False,
        context_tokens=max(1, len(prompt) // 4),
    )
    payload["format"] = schema
    # Merge rather than replace so a num_ctx chosen above survives.
    payload["options"] = {**payload.get("options", {}), "temperature": 0}
    content = post_json(
        _endpoint(model), payload, provider=PROVIDER, timeout=timeout
    ).get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"ollama returned unparseable classifier JSON: {content[:200]}", provider=PROVIDER
        ) from exc


def stream_ollama(
    model: dict[str, Any],
    messages: list[dict[str, str]],
    timeout: int = 60,
    context_tokens: int = 0,
) -> Iterator[str]:
    for line in stream_lines(
        _endpoint(model),
        _payload(model, messages, stream=True, context_tokens=context_tokens),
        provider=PROVIDER,
        timeout=timeout,
    ):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # A partial frame mid-stream is not worth aborting a response over.
            continue
        if data.get("error"):
            raise ProviderError(f"ollama stream error: {data['error']}", provider=PROVIDER)
        chunk = data.get("message", {}).get("content", "")
        if chunk:
            yield chunk


def _endpoint(model: dict[str, Any]) -> str:
    return str(model.get("endpoint") or DEFAULT_ENDPOINT)


def _payload(
    model: dict[str, Any],
    messages: list[dict[str, str]],
    stream: bool,
    context_tokens: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.get("model", model.get("id")),
        "messages": messages,
        "stream": stream,
    }
    num_ctx = resolve_num_ctx(model, context_tokens)
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    return payload
