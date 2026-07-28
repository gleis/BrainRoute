"""Provider dispatch.

Adapters speak in chat messages, not flattened strings, so a multi-turn
conversation or a system prompt survives the trip to the provider. Callers may
still pass a plain string for the common single-turn case.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from .http import DEFAULT_RETRIES, ProviderError, with_retries
from .ollama import call_ollama, stream_ollama
from .openai import call_openai, stream_openai
from .openai_compatible import call_openai_compatible, stream_openai_compatible


VALID_ROLES = {"system", "user", "assistant", "tool", "developer"}

_GENERATORS = {
    "ollama": call_ollama,
    "openai": call_openai,
    "openai_compatible": call_openai_compatible,
}

_STREAMERS = {
    "ollama": stream_ollama,
    "openai": stream_openai,
    "openai_compatible": stream_openai_compatible,
}


def to_messages(value: str | Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalise a prompt or message list into validated chat messages."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        if isinstance(content, list):
            # Tolerate OpenAI-style content parts by flattening their text.
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        messages.append({
            "role": role if role in VALID_ROLES else "user",
            "content": str(content),
        })
    return messages or [{"role": "user", "content": ""}]


def flatten(messages: Sequence[dict[str, str]]) -> str:
    """Render messages as text, for classification and token estimation only."""
    return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)


# Providers whose adapters accept a context hint. Cloud APIs size their own
# context; only local runtimes need to be told what window to allocate.
_CONTEXT_AWARE = {"ollama"}


def generate(
    model: dict[str, Any],
    prompt: str | Sequence[dict[str, Any]],
    timeout: int = 60,
    retries: int = DEFAULT_RETRIES,
    context_tokens: int = 0,
) -> str:
    provider = model.get("provider")
    handler = _GENERATORS.get(str(provider))
    if handler is None:
        raise ProviderError(
            f"Provider '{provider}' is not executable yet", provider=str(provider)
        )
    messages = to_messages(prompt)
    kwargs = _context_kwargs(str(provider), context_tokens)
    return with_retries(
        lambda: handler(model, messages, timeout=timeout, **kwargs),
        retries=retries,
        provider=str(provider),
    )


def stream_generate(
    model: dict[str, Any],
    prompt: str | Sequence[dict[str, Any]],
    timeout: int = 60,
    context_tokens: int = 0,
) -> Iterator[str]:
    """Stream tokens from a provider.

    Streaming is deliberately not retried: once any bytes have reached the
    client, a silent retry would duplicate output. Failures before the first
    chunk are handled by the caller's fallback chain.
    """
    provider = str(model.get("provider"))
    messages = to_messages(prompt)
    streamer = _STREAMERS.get(provider)
    if streamer is not None:
        yield from streamer(
            model, messages, timeout=timeout, **_context_kwargs(provider, context_tokens)
        )
        return
    # Providers without a streaming adapter still work, as a single chunk.
    yield generate(model, messages, timeout=timeout, context_tokens=context_tokens)


def _context_kwargs(provider: str, context_tokens: int) -> dict[str, Any]:
    return {"context_tokens": context_tokens} if provider in _CONTEXT_AWARE else {}
