"""Generic OpenAI-compatible chat-completions adapter.

The ``/v1/chat/completions`` shape is a de facto standard, so one adapter
reaches any endpoint that speaks it: self-hosted vLLM, llama.cpp, LM Studio,
managed inference gateways, corporate proxies, or a second ModelRoute.

Unlike the first-party OpenAI adapter, nothing here is hard-coded to a vendor.
A model declares where to send the request and which environment variable holds
its credential, so adding a provider is a config change rather than new code.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from .http import ProviderError, post_json, stream_lines


PROVIDER = "openai_compatible"
DEFAULT_API_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"


def call_openai_compatible(
    model: dict[str, Any], messages: list[dict[str, str]], timeout: int = 60
) -> str:
    data = post_json(
        _endpoint(model),
        _payload(model, messages, stream=False),
        provider=_label(model),
        headers=_headers(model),
        timeout=timeout,
    )
    _raise_for_error(data, model)
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def stream_openai_compatible(
    model: dict[str, Any], messages: list[dict[str, str]], timeout: int = 60
) -> Iterator[str]:
    for line in stream_lines(
        _endpoint(model),
        _payload(model, messages, stream=True),
        provider=_label(model),
        headers=_headers(model),
        timeout=timeout,
    ):
        # SSE comments are keepalives on several implementations.
        if line.startswith(":") or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        try:
            event = json.loads(body)
        except json.JSONDecodeError:
            continue
        _raise_for_error(event, model)
        for choice in event.get("choices", []):
            chunk = (choice.get("delta") or {}).get("content")
            if chunk:
                yield chunk


def _endpoint(model: dict[str, Any]) -> str:
    endpoint = str(model.get("endpoint") or "").strip()
    if not endpoint:
        raise ProviderError(
            f"Model '{model.get('id')}' needs an 'endpoint' to reach an "
            "OpenAI-compatible provider",
            provider=_label(model),
            status=400,
        )
    return endpoint


def _headers(model: dict[str, Any]) -> dict[str, str]:
    """Authorization plus any static headers the endpoint requires.

    A missing credential is not retryable - no amount of backoff produces one.
    An endpoint that needs no auth simply declares no ``api_key_env``.
    """
    headers = {str(key): str(value) for key, value in (model.get("headers") or {}).items()}
    variable = str(model.get("api_key_env") or DEFAULT_API_KEY_ENV)
    api_key = os.environ.get(variable)
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")
    elif model.get("requires_api_key", True):
        raise ProviderError(
            f"{variable} is required for model '{model.get('id')}'",
            provider=_label(model),
            status=401,
        )
    return headers


def _payload(
    model: dict[str, Any], messages: list[dict[str, str]], stream: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.get("model", model.get("id")),
        "messages": messages,
    }
    if stream:
        payload["stream"] = True
    if model.get("max_output_tokens"):
        payload["max_tokens"] = int(model["max_output_tokens"])
    # Escape hatch for endpoint-specific knobs without new code.
    payload.update(model.get("extra_body") or {})
    return payload


def _raise_for_error(data: dict[str, Any], model: dict[str, Any]) -> None:
    error = data.get("error")
    if not error:
        return
    message = error.get("message", error) if isinstance(error, dict) else error
    raise ProviderError(f"{_label(model)} error: {message}", provider=_label(model))


def _label(model: dict[str, Any]) -> str:
    """Prefer a human name in errors so multi-endpoint setups stay debuggable."""
    return str(model.get("provider_label") or PROVIDER)
