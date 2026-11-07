"""HTTP gateway: local UI, admin API, and an OpenAI-compatible endpoint.

Requests are dispatched from a routing table rather than a chain of path
comparisons, every handler runs inside one error boundary so a bug returns JSON
instead of a dropped connection, and every response carries a request id that
also appears in the telemetry record for that request.
"""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .catalog import discover_models
from .config import load_config
from .evals import run_evals
from .execution import execute_with_fallback, required_context, stream_with_metrics
from .providers import ProviderError, flatten, provider_health, to_messages
from .reliability import snapshot as reliability_snapshot
from .router import NoRouteAvailable, find_model, route
from .postures import describe as describe_postures
from .settings import (
    clear_policy_overrides,
    load_settings,
    load_stored_settings,
    save_settings,
    set_posture,
    update_model,
)
from .store import append_message, dashboard, get_session, list_sessions, prune
from .telemetry import read_events, write_event


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 2 * 1024 * 1024
AUTO_MODEL = "brainroute-auto"

# Endpoints that change state or spend money. When an API key is configured
# these require it, so an unauthenticated caller cannot rewrite routing policy
# or enable an expensive model.
PROTECTED_PATHS = {
    "/api/settings",
    "/api/postures",
    "/api/models",
    "/api/ask",
    "/api/chat/stream",
    "/api/compare",
    "/api/maintenance/prune",
}


class RateLimiter:
    """Fixed-window request limiter keyed by client address."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(client, deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            if len(self._hits) > 1024:
                self._prune_locked(now)
            return True

    def _prune_locked(self, now: float) -> None:
        stale = [key for key, window in self._hits.items() if not window or now - window[-1] > 120.0]
        for key in stale:
            self._hits.pop(key, None)


_rate_limiter = RateLimiter(load_settings()["gateway"].get("rate_limit_per_minute", 120))


class BrainRouteHandler(BaseHTTPRequestHandler):
    server_version = "BrainRoute/1.0"
    protocol_version = "HTTP/1.1"

    # ---- dispatch ---------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        self.request_id = uuid.uuid4().hex[:16]
        self.path_only = urlparse(self.path).path

        # Only the API surface is metered. Loading the UI costs several static
        # requests, and throttling those would break the page that is meant to
        # show you the throttling.
        if self.path_only.startswith(("/api/", "/v1/")):
            client = self.client_address[0] if self.client_address else "unknown"
            if not _rate_limiter.allow(client):
                self._json(
                    {"error": {"message": "Rate limit exceeded", "type": "rate_limit"}}, status=429
                )
                return
        if method == "POST" and self._body_too_large():
            return
        if self._requires_auth() and not self._authorised():
            self._json({"error": {"message": "Unauthorized", "type": "auth_error"}}, status=401)
            return

        handler = _ROUTES.get((method, self.path_only))
        if handler is None:
            handler = self._prefix_handler(method)
        if handler is None:
            if method == "GET":
                self._static()
            else:
                self._json({"error": {"message": "Not found", "type": "not_found"}}, status=404)
            return

        try:
            handler(self)
        except NoRouteAvailable as exc:
            self._json({"error": {"message": str(exc), "type": "no_route"}}, status=409)
        except ValueError as exc:
            self._json({"error": {"message": str(exc), "type": "invalid_request"}}, status=400)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001 - the boundary must not leak tracebacks
            write_event({"event": "error", "request_id": self.request_id, "path": self.path_only, "error": str(exc)})
            self._json({"error": {"message": f"Internal error: {exc}", "type": "internal"}}, status=500)

    def _prefix_handler(self, method: str) -> Callable[[BrainRouteHandler], None] | None:
        for (route_method, prefix), handler in _PREFIX_ROUTES.items():
            if method == route_method and self.path_only.startswith(prefix):
                return handler
        return None

    def log_message(self, format: str, *args: Any) -> None:
        return

    # ---- read endpoints ---------------------------------------------------

    def _healthz(self) -> None:
        self._json({"ok": True, "service": "brainroute", "version": self.server_version})

    def _api_config(self) -> None:
        config = load_config(discover=True)
        settings = load_settings()
        self._json({
            "profiles": sorted(config.profiles),
            "default_profile": settings["policy"].get("posture_default_profile") or config.default_profile,
            "configured_default_profile": config.default_profile,
            "models": [_public_model(model) for model in config.models],
            "postures": describe_postures(),
            "pinned_policy": sorted(load_stored_settings().get("policy", {})),
            "settings": settings,
        })

    def _api_postures(self) -> None:
        settings = load_settings()
        self._json({
            "active": settings["policy"]["posture"],
            "postures": describe_postures(),
            "pinned_policy": sorted(load_stored_settings().get("policy", {})),
        })

    def _api_set_posture(self) -> None:
        payload = self._read_json()
        name = str(payload.get("posture", ""))
        known = {item["id"] for item in describe_postures()}
        if name not in known:
            raise ValueError(f"Unknown posture '{name}'. Available: {', '.join(sorted(known))}")
        if payload.get("reset_overrides"):
            clear_policy_overrides()
        settings = set_posture(name)
        write_event({"event": "posture", "request_id": self.request_id, "posture": name})
        self._json({"ok": True, "settings": settings})

    def _api_catalog(self) -> None:
        self._json(discover_models())

    def _api_health(self) -> None:
        config = load_config(discover=True)
        self._json({"providers": provider_health(config.models)})

    def _api_reliability(self) -> None:
        self._json({"models": [item.as_dict() for item in reliability_snapshot().values()]})

    def _api_telemetry(self) -> None:
        self._json({"events": read_events(limit=50)})

    def _api_dashboard(self) -> None:
        self._json(dashboard())

    def _api_evals(self) -> None:
        results = run_evals(load_config(discover=True))
        self._json({
            "results": [result.as_dict() for result in results],
            "passed": sum(1 for result in results if result.passed),
            "total": len(results),
        })

    def _api_sessions(self) -> None:
        self._json({"sessions": list_sessions()})

    def _api_session(self) -> None:
        self._json(get_session(self.path_only.rsplit("/", 1)[-1]))

    def _v1_models(self) -> None:
        config = load_config(discover=True)
        data = [
            {
                "id": model["id"],
                "object": "model",
                "owned_by": model.get("provider", "brainroute"),
                "context_window": model.get("max_context"),
            }
            for model in config.models
            if model.get("enabled")
        ]
        data.insert(0, {"id": AUTO_MODEL, "object": "model", "owned_by": "brainroute"})
        self._json({"object": "list", "data": data})

    # ---- write endpoints --------------------------------------------------

    def _api_route(self) -> None:
        payload = self._read_json()
        prompt = _prompt_from(payload)
        decision = route(
            prompt, load_config(discover=True), profile_name=payload.get("profile"),
            request_id=self.request_id, overrides=_overrides_from(payload),
        )
        write_event({
            "event": "route",
            "request_id": self.request_id,
            "profile": decision.profile_name,
            "selected_model": decision.selected.model["id"],
            "decision": decision.as_dict(),
        })
        self._json(decision.as_dict())

    def _api_ask(self) -> None:
        payload = self._read_json()
        messages = _messages_from(payload)
        prompt = flatten(messages)
        config = load_config(discover=True)
        decision = route(
            prompt, config, profile_name=payload.get("profile"),
            request_id=self.request_id, overrides=_overrides_from(payload),
        )
        model = find_model(config, payload["model"]) if payload.get("model") else decision.selected.model
        execute = bool(payload.get("execute"))

        response: dict[str, Any] = {
            "request_id": self.request_id,
            "decision": {**decision.as_dict(), "recommended_model": model["id"]},
            "executed": execute,
            "output": "",
            "error": None,
        }
        if execute:
            result = execute_with_fallback(model, decision, messages)
            response["output"] = result.output
            response["error"] = result.error
            response["fallback_attempted"] = result.fallback_attempted
            response["attempts"] = [item.as_dict() for item in result.attempts]
            response["decision"]["recommended_model"] = result.model["id"]

        session_id = payload.get("session_id")
        if session_id:
            append_message(session_id, "user", prompt, {"profile": decision.profile_name})
            if response["output"]:
                append_message(
                    session_id, "assistant", response["output"],
                    {"model": response["decision"]["recommended_model"], "request_id": self.request_id},
                )
        write_event({
            "event": "ask",
            "request_id": self.request_id,
            "selected_model": response["decision"]["recommended_model"],
            "executed": execute,
            "error": response["error"],
            "response_chars": len(response["output"]),
        })
        self._json(response, status=200 if not response["error"] else 502)

    def _api_feedback(self) -> None:
        payload = self._read_json()
        write_event({
            "event": "feedback",
            "request_id": payload.get("request_id") or self.request_id,
            "rating": payload.get("rating"),
            "selected_model": payload.get("model"),
            "profile": payload.get("profile"),
            "task_type": payload.get("task_type"),
        })
        # Feedback feeds the quality factor, so make it visible immediately.
        from . import reliability

        reliability.invalidate()
        self._json({"ok": True})

    def _api_models(self) -> None:
        payload = self._read_json()
        model_id = str(payload.get("id", ""))
        if not model_id:
            raise ValueError("Model id is required")
        update: dict[str, Any] = {"enabled": bool(payload.get("enabled"))}
        for field in ("quality_score", "speed_score", "privacy_score", "cost_score", "max_context"):
            if field in payload:
                update[field] = payload[field]
        self._json({"ok": True, "settings": update_model(model_id, update)})

    def _api_settings(self) -> None:
        settings = save_settings(self._read_json())
        _rate_limiter.per_minute = int(settings["gateway"].get("rate_limit_per_minute", 120))
        self._json({"ok": True, "settings": settings})

    def _api_prune(self) -> None:
        self._json({"ok": True, **prune()})

    def _api_compare(self) -> None:
        payload = self._read_json()
        messages = _messages_from(payload)
        config = load_config(discover=True)
        decision = route(
            flatten(messages), config, profile_name=payload.get("profile"),
            request_id=self.request_id, overrides=_overrides_from(payload),
        )
        model_ids = payload.get("models") or [item.model["id"] for item in decision.ranked[:2]]
        comparisons = []
        for model_id in model_ids[:2]:
            model = find_model(config, model_id)
            item: dict[str, Any] = {
                "model": model_id,
                "provider": model.get("provider"),
                "output": "",
                "error": None,
            }
            if payload.get("execute"):
                result = execute_with_fallback(model, decision, messages)
                item.update({
                    "model": result.model["id"],
                    "output": result.output,
                    "error": result.error,
                    "fallback_attempted": result.fallback_attempted,
                })
            comparisons.append(item)
        self._json({"decision": decision.as_dict(), "comparisons": comparisons})

    # ---- streaming --------------------------------------------------------

    def _api_chat_stream(self) -> None:
        payload = self._read_json()
        messages = _messages_from(payload)
        prompt = flatten(messages)
        config = load_config(discover=True)
        decision = route(
            prompt, config, profile_name=payload.get("profile"),
            request_id=self.request_id, overrides=_overrides_from(payload),
        )
        model = find_model(config, payload["model"]) if payload.get("model") else decision.selected.model
        session_id = str(payload.get("session_id", ""))
        if session_id:
            append_message(session_id, "user", prompt, {"profile": decision.profile_name})

        self._begin_stream()
        self._sse("route", {"decision": {**decision.as_dict(), "recommended_model": model["id"]}})

        candidates = _stream_candidates(model, decision)
        needed = required_context(messages, decision)
        for candidate in candidates:
            output: list[str] = []
            try:
                for chunk in stream_with_metrics(candidate, messages, needed):
                    output.append(chunk)
                    self._sse("chunk", {"text": chunk})
            except (BrokenPipeError, ConnectionResetError):
                return
            except ProviderError as exc:
                if output:
                    # Bytes already reached the client; switching models now
                    # would splice two different answers together.
                    self._sse("error", {"error": str(exc)})
                    return
                remaining = candidates[candidates.index(candidate) + 1:]
                self._sse("fallback", {
                    "from": candidate["id"],
                    "to": remaining[0]["id"] if remaining else None,
                    "reason": str(exc),
                })
                continue

            text = "".join(output)
            if session_id:
                append_message(session_id, "assistant", text, {"model": candidate["id"], "request_id": self.request_id})
            write_event({
                "event": "stream",
                "request_id": self.request_id,
                "selected_model": candidate["id"],
                "response_chars": len(text),
                "profile": decision.profile_name,
            })
            self._sse("done", {"model": candidate["id"], "response_chars": len(text)})
            return

        self._sse("error", {"error": "every candidate model failed"})

    def _v1_chat(self) -> None:
        payload = self._read_json()
        messages = to_messages(payload.get("messages") or [])
        config = load_config(discover=True)
        decision = route(
            flatten(messages), config,
            profile_name=payload.get("brainroute_profile"),
            request_id=self.request_id,
            overrides={"allow_cloud": bool(payload.get("brainroute_allow_cloud"))},
        )
        requested = payload.get("model")
        model = (
            find_model(config, requested)
            if requested and requested != AUTO_MODEL
            else decision.selected.model
        )
        if payload.get("stream"):
            self._v1_chat_stream(model, decision, messages)
            return

        result = execute_with_fallback(model, decision, messages)
        if result.error:
            self._json(
                {"error": {"message": result.error, "type": "provider_error", "request_id": self.request_id}},
                status=502,
            )
            return
        self._json({
            "id": f"brainroute-{self.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model["id"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": result.output},
            }],
            "brainroute": {
                "decision": decision.as_dict(),
                "fallback_attempted": result.fallback_attempted,
                "attempts": [item.as_dict() for item in result.attempts],
            },
        })

    def _v1_chat_stream(self, model: dict[str, Any], decision: Any, messages: list[dict[str, str]]) -> None:
        """Emit OpenAI-compatible streaming chunks so standard clients work."""
        completion_id = f"brainroute-{self.request_id}"
        needed = required_context(messages, decision)
        self._begin_stream()
        for candidate in _stream_candidates(model, decision):
            emitted = False
            try:
                for chunk in stream_with_metrics(candidate, messages, needed):
                    if not emitted:
                        self._sse_data(_chunk_frame(completion_id, candidate["id"], {"role": "assistant"}))
                        emitted = True
                    self._sse_data(_chunk_frame(completion_id, candidate["id"], {"content": chunk}))
            except (BrokenPipeError, ConnectionResetError):
                return
            except ProviderError as exc:
                if emitted:
                    self._sse_data(_chunk_frame(completion_id, candidate["id"], {}, finish_reason="stop"))
                    self._sse_raw("[DONE]")
                    write_event({"event": "stream", "request_id": self.request_id, "error": str(exc)})
                    return
                continue

            self._sse_data(_chunk_frame(completion_id, candidate["id"], {}, finish_reason="stop"))
            self._sse_raw("[DONE]")
            write_event({
                "event": "stream",
                "request_id": self.request_id,
                "selected_model": candidate["id"],
                "gateway": "v1",
            })
            return
        self._sse_data({"error": {"message": "every candidate model failed", "type": "provider_error"}})
        self._sse_raw("[DONE]")

    # ---- transport helpers ------------------------------------------------

    def _begin_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Request-Id", self.request_id)
        self.send_header("Connection", "close")
        self.close_connection = True
        self._security_headers()
        self.end_headers()

    def _sse(self, event: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), default=str)
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _sse_data(self, payload: dict[str, Any]) -> None:
        self._sse_raw(json.dumps(payload, separators=(",", ":"), default=str))

    def _sse_raw(self, data: str) -> None:
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _body_too_large(self) -> bool:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self._json({"error": {"message": "Invalid Content-Length", "type": "invalid_request"}}, status=400)
            return True
        if length > MAX_BODY_BYTES:
            # The body is never read, so the connection cannot be reused - the
            # unread bytes would be parsed as the next request on a keep-alive
            # connection.
            self.close_connection = True
            self._json({"error": {"message": "Request body exceeds 2 MiB", "type": "too_large"}}, status=413)
            return True
        return False

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-Id", getattr(self, "request_id", "-"))
        self._security_headers()
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _static(self) -> None:
        requested = self.path_only.lstrip("/") or "index.html"
        target = (STATIC_DIR / requested).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            target = STATIC_DIR / "index.html"
        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self._security_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- auth -------------------------------------------------------------

    def _requires_auth(self) -> bool:
        if not os.environ.get("BRAINROUTE_API_KEY"):
            return False
        if self.path_only.startswith("/v1/"):
            return True
        if not load_settings()["gateway"].get("protect_admin_api", True):
            return False
        return self.path_only in PROTECTED_PATHS

    def _authorised(self) -> bool:
        expected = os.environ.get("BRAINROUTE_API_KEY", "")
        provided = self.headers.get("Authorization", "")
        # Constant-time comparison keeps the key out of reach of timing attacks.
        return hmac.compare_digest(provided, f"Bearer {expected}")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")


_ROUTES: dict[tuple[str, str], Callable[[BrainRouteHandler], None]] = {
    ("GET", "/healthz"): BrainRouteHandler._healthz,
    ("GET", "/api/config"): BrainRouteHandler._api_config,
    ("GET", "/api/catalog"): BrainRouteHandler._api_catalog,
    ("GET", "/api/health"): BrainRouteHandler._api_health,
    ("GET", "/api/reliability"): BrainRouteHandler._api_reliability,
    ("GET", "/api/postures"): BrainRouteHandler._api_postures,
    ("GET", "/api/dashboard"): BrainRouteHandler._api_dashboard,
    ("GET", "/api/evals"): BrainRouteHandler._api_evals,
    ("GET", "/api/sessions"): BrainRouteHandler._api_sessions,
    ("GET", "/v1/models"): BrainRouteHandler._v1_models,
    ("POST", "/api/route"): BrainRouteHandler._api_route,
    ("POST", "/api/ask"): BrainRouteHandler._api_ask,
    ("POST", "/api/chat/stream"): BrainRouteHandler._api_chat_stream,
    ("POST", "/api/compare"): BrainRouteHandler._api_compare,
    ("POST", "/api/feedback"): BrainRouteHandler._api_feedback,
    ("POST", "/api/models"): BrainRouteHandler._api_models,
    ("POST", "/api/settings"): BrainRouteHandler._api_settings,
    ("POST", "/api/postures"): BrainRouteHandler._api_set_posture,
    ("POST", "/api/maintenance/prune"): BrainRouteHandler._api_prune,
    ("POST", "/v1/chat/completions"): BrainRouteHandler._v1_chat,
}

_PREFIX_ROUTES: dict[tuple[str, str], Callable[[BrainRouteHandler], None]] = {
    ("GET", "/api/telemetry"): BrainRouteHandler._api_telemetry,
    ("GET", "/api/sessions/"): BrainRouteHandler._api_session,
}


def _stream_candidates(model: dict[str, Any], decision: Any) -> list[dict[str, Any]]:
    ordered = [model]
    seen = {model["id"]}
    for scored in getattr(decision, "fallback_chain", ()):
        if scored.model["id"] not in seen:
            ordered.append(scored.model)
            seen.add(scored.model["id"])
    return ordered


def _chunk_frame(
    completion_id: str,
    model_id: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _overrides_from(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-request policy opt-ins. Postures decide whether these are honoured."""
    return {"allow_cloud": bool(payload.get("allow_cloud"))}


def _prompt_from(payload: dict[str, Any]) -> str:
    if payload.get("messages"):
        return flatten(to_messages(payload["messages"]))
    return str(payload.get("prompt", ""))


def _messages_from(payload: dict[str, Any]) -> list[dict[str, str]]:
    if payload.get("messages"):
        return to_messages(payload["messages"])
    return to_messages(str(payload.get("prompt", "")))


def _public_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "provider": model.get("provider"),
        "model": model.get("model"),
        "local": model.get("local"),
        "enabled": model.get("enabled"),
        "discovered": model.get("discovered", False),
        "strengths": model.get("strengths", []),
        "weaknesses": model.get("weaknesses", []),
        "quality_score": model.get("quality_score"),
        "speed_score": model.get("speed_score"),
        "cost_score": model.get("cost_score"),
        "privacy_score": model.get("privacy_score"),
        "max_context": model.get("max_context"),
        "context_source": model.get("context_source"),
        "input_price_per_mtok": model.get("input_price_per_mtok"),
        "output_price_per_mtok": model.get("output_price_per_mtok"),
        "pricing_known": model.get("pricing_known", False),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainroute-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), BrainRouteHandler)
    server.daemon_threads = True
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("BRAINROUTE_API_KEY"):
        print("Warning: binding beyond loopback without BRAINROUTE_API_KEY set.")
    print(f"ModelRoute running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
