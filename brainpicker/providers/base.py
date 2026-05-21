from __future__ import annotations

from typing import Any

from .ollama import call_ollama
from .openai import call_openai


class ProviderError(RuntimeError):
    pass


def generate(model: dict[str, Any], prompt: str, timeout: int = 60) -> str:
    provider = model.get("provider")
    if provider == "ollama":
        return call_ollama(model, prompt, timeout=timeout)
    if provider == "openai":
        return call_openai(model, prompt, timeout=timeout)
    raise ProviderError(f"Provider '{provider}' is not executable yet")

