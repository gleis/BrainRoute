from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Iterator

from .providers import ProviderError, generate, stream_generate
from .router import RouteDecision
from .store import record_run


@dataclass(frozen=True)
class ExecutionResult:
    output: str
    model: dict[str, Any]
    fallback_attempted: str | None = None
    error: str | None = None


def execute_with_fallback(model: dict[str, Any], decision: RouteDecision, prompt: str) -> ExecutionResult:
    try:
        return ExecutionResult(output=_timed_generate(model, prompt), model=model)
    except ProviderError as exc:
        fallback = decision.fallback.model if decision.fallback else None
        if not fallback or fallback["id"] == model["id"]:
            return ExecutionResult(output="", model=model, error=str(exc))
        try:
            return ExecutionResult(
                output=_timed_generate(fallback, prompt),
                model=fallback,
                fallback_attempted=fallback["id"],
            )
        except ProviderError as fallback_exc:
            return ExecutionResult(
                output="",
                model=model,
                fallback_attempted=fallback["id"],
                error=f"{exc}; fallback failed: {fallback_exc}",
            )


def stream_with_metrics(model: dict[str, Any], prompt: str) -> Iterator[str]:
    started = monotonic()
    output: list[str] = []
    ok = False
    try:
        for chunk in stream_generate(model, prompt):
            output.append(chunk)
            yield chunk
        ok = True
    finally:
        _record(model, prompt, "".join(output), started, ok)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_cost(model: dict[str, Any], prompt: str, output: str) -> float:
    return round(
        estimate_tokens(prompt) * float(model.get("input_cost_per_token", 0))
        + estimate_tokens(output) * float(model.get("output_cost_per_token", 0)),
        8,
    )


def _timed_generate(model: dict[str, Any], prompt: str) -> str:
    started = monotonic()
    output = ""
    ok = False
    try:
        output = generate(model, prompt)
        ok = True
        return output
    finally:
        _record(model, prompt, output, started, ok)


def _record(model: dict[str, Any], prompt: str, output: str, started: float, ok: bool) -> None:
    record_run({
        "model_id": model.get("id"),
        "provider": model.get("provider"),
        "latency_ms": round((monotonic() - started) * 1000),
        "input_tokens": estimate_tokens(prompt),
        "output_tokens": estimate_tokens(output),
        "estimated_cost_usd": estimate_cost(model, prompt, output),
        "ok": ok,
    })

