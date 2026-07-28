"""Shared HTTP plumbing for provider adapters.

Centralises the two things every adapter needs and that were previously
duplicated (and inconsistent) across them: turning transport failures into a
typed error that says whether a retry is worthwhile, and reading a streaming
response line by line without buffering the whole body.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator


# Status codes worth retrying: rate limits and transient server-side failures.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
DEFAULT_RETRIES = 2
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0


class ProviderError(RuntimeError):
    """A provider call failed. ``retryable`` drives both retry and fallback."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "provider": self.provider,
            "status": self.status,
            "retryable": self.retryable,
        }


def build_request(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    request = build_request(url, payload, headers)
    with _open(request, provider=provider, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"{provider} returned a non-JSON response: {body[:200]}", provider=provider
        ) from exc


def get_json(url: str, *, provider: str, headers: dict[str, str] | None = None, timeout: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with _open(request, provider=provider, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"{provider} returned a non-JSON response: {body[:200]}", provider=provider
        ) from exc


def stream_lines(
    url: str,
    payload: dict[str, Any],
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> Iterator[str]:
    request = build_request(url, payload, headers)
    with _open(request, provider=provider, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                yield line


def with_retries(
    operation: Callable[[], Any],
    *,
    retries: int = DEFAULT_RETRIES,
    provider: str = "provider",
) -> Any:
    """Retry a provider call on transient failures with jittered backoff.

    Only errors flagged retryable are retried; a 401 or a malformed request
    will fail the same way every time and retrying it just adds latency.
    """
    last_error: ProviderError | None = None
    for attempt in range(retries + 1):
        try:
            return operation()
        except ProviderError as exc:
            last_error = exc
            if not exc.retryable or attempt == retries:
                raise
            time.sleep(_backoff(attempt))
    # Unreachable, but keeps the type checker and future readers honest.
    raise last_error or ProviderError(f"{provider} failed", provider=provider)


def _backoff(attempt: int) -> float:
    delay = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** attempt))
    return delay * (0.5 + random.random() / 2)


def _open(request: urllib.request.Request, *, provider: str, timeout: int):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = _safe_read(exc)
        raise ProviderError(
            f"{provider} request failed: {exc.code} {detail}",
            provider=provider,
            status=exc.code,
            retryable=exc.code in RETRYABLE_STATUS,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Connection refused, DNS failure, and timeouts are all transient from
        # the router's point of view; the provider may simply be restarting.
        raise ProviderError(
            f"{provider} request failed: {exc}",
            provider=provider,
            retryable=True,
        ) from exc


def _safe_read(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:400]
    except Exception:  # noqa: BLE001 - error bodies are best-effort
        return exc.reason if isinstance(exc.reason, str) else ""
