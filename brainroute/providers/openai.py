from __future__ import annotations

import json
import os
from typing import Any, Iterator

from .http import ProviderError, post_json, stream_lines


PROVIDER = "openai"
RESPONSES_URL = "https://api.openai.com/v1/responses"


def call_openai(model: dict[str, Any], messages: list[dict[str, str]], timeout: int = 60) -> str:
    data = post_json(
        _endpoint(model),
        _payload(model, messages, stream=False),
        provider=PROVIDER,
        headers=_headers(),
        timeout=timeout,
    )
    return _extract_output_text(data)


def stream_openai(
    model: dict[str, Any], messages: list[dict[str, str]], timeout: int = 60
) -> Iterator[str]:
    """Stream via the Responses API's typed server-sent events."""
    for line in stream_lines(
        _endpoint(model),
        _payload(model, messages, stream=True),
        provider=PROVIDER,
        headers=_headers(),
        timeout=timeout,
    ):
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        try:
            event = json.loads(body)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                yield delta
        elif event_type == "error":
            message = event.get("error", {}).get("message", "unknown streaming error")
            raise ProviderError(f"openai stream error: {message}", provider=PROVIDER)


def _endpoint(model: dict[str, Any]) -> str:
    return str(model.get("endpoint") or RESPONSES_URL)


def _headers() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Not retryable: no amount of backoff produces a credential.
        raise ProviderError(
            "OPENAI_API_KEY is required for OpenAI models", provider=PROVIDER, status=401
        )
    return {"Authorization": f"Bearer {api_key}"}


def _payload(model: dict[str, Any], messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
    """Map chat messages onto the Responses API input format.

    The previous adapter flattened the whole conversation into one string,
    which discarded role structure and any system prompt.
    """
    system_parts = [item["content"] for item in messages if item.get("role") == "system"]
    conversation = [
        {
            "role": item.get("role", "user"),
            "content": [{"type": _content_type(item), "text": item.get("content", "")}],
        }
        for item in messages
        if item.get("role") != "system"
    ]
    payload: dict[str, Any] = {
        "model": model.get("model", model.get("id")),
        "input": conversation or [
            {"role": "user", "content": [{"type": "input_text", "text": ""}]}
        ],
    }
    if system_parts:
        payload["instructions"] = "\n\n".join(system_parts)
    if stream:
        payload["stream"] = True
    if model.get("max_output_tokens"):
        payload["max_output_tokens"] = int(model["max_output_tokens"])
    return payload


def _content_type(message: dict[str, str]) -> str:
    return "output_text" if message.get("role") == "assistant" else "input_text"


def _extract_output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and "text" in content:
                parts.append(str(content["text"]))
    return "\n".join(parts)
