from __future__ import annotations

from typing import Any

from .ollama import call_ollama, stream_ollama
from .openai import call_openai
from .openrouter import call_openrouter


class ProviderError(RuntimeError):
    pass


def generate(model: dict[str, Any], prompt: str, timeout: int = 60) -> str:
    provider = model.get("provider")
    try:
        if provider == "ollama":
            return call_ollama(model, prompt, timeout=timeout)
        if provider == "openai":
            return call_openai(model, prompt, timeout=timeout)
        if provider == "openrouter":
            return call_openrouter(model, prompt, timeout=timeout)
    except Exception as exc:
        raise ProviderError(str(exc)) from exc
    raise ProviderError(f"Provider '{provider}' is not executable yet")


def stream_generate(model: dict[str, Any], prompt: str, timeout: int = 60):
    provider = model.get("provider")
    try:
        if provider == "ollama":
            yield from stream_ollama(model, prompt, timeout=timeout)
            return
        yield generate(model, prompt, timeout=timeout)
    except Exception as exc:
        raise ProviderError(str(exc)) from exc
