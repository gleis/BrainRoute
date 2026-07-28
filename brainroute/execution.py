"""Execution and fallback.

Routing picks a model; this module makes sure a request still gets an answer
when that model is unavailable. Every attempt is recorded, successful or not,
because those records are the evidence the router learns from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Iterator, Sequence

from . import reliability
from .classifier import estimate_tokens
from .pricing import ASSUMED_OUTPUT_TOKENS, estimate_request_cost
from .providers import ProviderError, flatten, generate, stream_generate, to_messages
from .router import RouteDecision
from .store import record_run


@dataclass(frozen=True)
class Attempt:
    model_id: str
    ok: bool
    latency_ms: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class ExecutionResult:
    output: str
    model: dict[str, Any]
    fallback_attempted: str | None = None
    error: str | None = None
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.get("id"),
            "output": self.output,
            "error": self.error,
            "fallback_attempted": self.fallback_attempted,
            "attempts": [item.as_dict() for item in self.attempts],
        }


def execute_with_fallback(
    model: dict[str, Any],
    decision: RouteDecision,
    prompt: str | Sequence[dict[str, Any]],
) -> ExecutionResult:
    """Try the selected model, then walk the ranked fallback chain.

    The previous implementation attempted exactly one fallback. With a chain,
    a single unhealthy provider no longer takes down requests that any other
    admitted model could have served.
    """
    messages = to_messages(prompt)
    candidates = _candidates(model, decision)
    needed = required_context(messages, decision)
    attempts: list[Attempt] = []
    errors: list[str] = []

    for index, candidate in enumerate(candidates):
        started = monotonic()
        try:
            output = _timed_generate(candidate, messages, needed)
        except ProviderError as exc:
            elapsed = _elapsed_ms(started)
            attempts.append(Attempt(candidate["id"], False, elapsed, str(exc)))
            errors.append(f"{candidate['id']}: {exc}")
            continue
        attempts.append(Attempt(candidate["id"], True, _elapsed_ms(started)))
        return ExecutionResult(
            output=output,
            model=candidate,
            fallback_attempted=candidate["id"] if index > 0 else None,
            attempts=tuple(attempts),
        )

    return ExecutionResult(
        output="",
        model=model,
        fallback_attempted=candidates[1]["id"] if len(candidates) > 1 else None,
        error="; ".join(errors) or "no candidate models were available",
        attempts=tuple(attempts),
    )


def stream_with_metrics(
    model: dict[str, Any],
    prompt: str | Sequence[dict[str, Any]],
    context_tokens: int | None = None,
) -> Iterator[str]:
    messages = to_messages(prompt)
    needed = context_tokens if context_tokens is not None else required_context(messages)
    started = monotonic()
    output: list[str] = []
    ok = False
    try:
        for chunk in stream_generate(model, messages, context_tokens=needed):
            output.append(chunk)
            yield chunk
        ok = True
    finally:
        _record(model, messages, "".join(output), started, ok)


def required_context(
    messages: list[dict[str, str]], decision: RouteDecision | None = None
) -> int:
    """Tokens the request needs end to end: prompt plus room for a reply.

    A local runtime allocates exactly this much rather than a model's ceiling,
    so a long prompt is not truncated and a short one costs no extra memory.
    The classifier's estimate is preferred when available, since it may account
    for content the raw text length does not.
    """
    prompt_tokens = estimate_tokens(flatten(messages))
    if decision is not None:
        prompt_tokens = max(prompt_tokens, decision.task.context_tokens)
    return prompt_tokens + ASSUMED_OUTPUT_TOKENS


def estimate_cost(
    model: dict[str, Any], prompt: str | Sequence[dict[str, Any]], output: str
) -> float:
    text = prompt if isinstance(prompt, str) else flatten(to_messages(prompt))
    return estimate_request_cost(model, estimate_tokens(text), estimate_tokens(output))


def _candidates(model: dict[str, Any], decision: RouteDecision) -> list[dict[str, Any]]:
    ordered = [model]
    seen = {model["id"]}
    for scored in decision.fallback_chain:
        candidate = scored.model
        if candidate["id"] not in seen:
            ordered.append(candidate)
            seen.add(candidate["id"])
    return ordered


def _timed_generate(
    model: dict[str, Any], messages: list[dict[str, str]], context_tokens: int = 0
) -> str:
    started = monotonic()
    output = ""
    ok = False
    try:
        output = generate(model, messages, context_tokens=context_tokens)
        ok = True
        return output
    finally:
        _record(model, messages, output, started, ok)


def _record(
    model: dict[str, Any],
    messages: list[dict[str, str]],
    output: str,
    started: float,
    ok: bool,
) -> None:
    prompt_text = flatten(messages)
    record_run({
        "model_id": model.get("id"),
        "provider": model.get("provider"),
        "latency_ms": _elapsed_ms(started),
        "input_tokens": estimate_tokens(prompt_text),
        "output_tokens": estimate_tokens(output),
        "estimated_cost_usd": estimate_request_cost(
            model, estimate_tokens(prompt_text), estimate_tokens(output)
        ),
        "ok": ok,
    })
    # A failure must be visible to the next routing decision immediately,
    # otherwise the breaker cannot trip within the cache TTL.
    if not ok:
        reliability.invalidate()


def _elapsed_ms(started: float) -> int:
    return round((monotonic() - started) * 1000)
