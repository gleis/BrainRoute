from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .catalog import discover_models
from .config import load_config
from .execution import execute_with_fallback, stream_with_metrics
from .evals import run_evals
from .providers import ProviderError, provider_health
from .router import find_model, route
from .settings import load_settings, save_settings, update_model
from .store import append_message, dashboard, get_session, list_sessions
from .telemetry import read_events, write_event


STATIC_DIR = Path(__file__).resolve().parent / "static"


class BrainRouteHandler(BaseHTTPRequestHandler):
    server_version = "BrainRoute/0.1"

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            if not self._api_auth():
                return
            config = load_config(discover=True)
            self._json({"object": "list", "data": [{"id": model["id"], "object": "model"} for model in config.models if model.get("enabled")]})
            return
        if self.path == "/api/config":
            config = load_config(discover=True)
            self._json({
                "profiles": sorted(config.profiles),
                "default_profile": config.default_profile,
                "models": [_public_model(model) for model in config.models],
                "settings": load_settings(),
            })
            return
        if self.path == "/api/catalog":
            self._json(discover_models())
            return
        if self.path == "/api/health":
            config = load_config(discover=True)
            self._json({"providers": provider_health(config.models)})
            return
        if self.path.startswith("/api/telemetry"):
            self._json({"events": read_events(limit=50)})
            return
        if self.path == "/api/dashboard":
            self._json(dashboard())
            return
        if self.path == "/api/evals":
            results = run_evals(load_config(discover=True))
            self._json({"results": [result.__dict__ for result in results]})
            return
        if self.path == "/api/sessions":
            self._json({"sessions": list_sessions()})
            return
        if self.path.startswith("/api/sessions/"):
            self._json(get_session(self.path.rsplit("/", 1)[-1]))
            return
        self._static()

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            if not self._api_auth():
                return
            self._v1_chat(self._read_json())
            return
        if self.path == "/api/route":
            payload = self._read_json()
            decision = route(str(payload.get("prompt", "")), load_config(discover=True), profile_name=payload.get("profile"))
            event = {"event": "route", "profile": decision.profile_name, "decision": decision.as_dict()}
            write_event(event)
            self._json(decision.as_dict())
            return

        if self.path == "/api/ask":
            payload = self._read_json()
            prompt = str(payload.get("prompt", ""))
            config = load_config(discover=True)
            decision = route(prompt, config, profile_name=payload.get("profile"))
            model = find_model(config, payload["model"]) if payload.get("model") else decision.selected.model
            execute = bool(payload.get("execute"))
            response: dict[str, Any] = {
                "decision": {**decision.as_dict(), "recommended_model": model["id"]},
                "executed": execute,
                "output": "",
                "error": None,
            }
            if execute:
                result = execute_with_fallback(model, decision, prompt)
                response["output"] = result.output
                response["error"] = result.error
                response["fallback_attempted"] = result.fallback_attempted
                response["decision"]["recommended_model"] = result.model["id"]
            if payload.get("session_id"):
                append_message(payload["session_id"], "user", prompt, {"profile": decision.profile_name})
                if response["output"]:
                    append_message(
                        payload["session_id"],
                        "assistant",
                        response["output"],
                        {"model": response["decision"]["recommended_model"]},
                    )
            write_event({
                "event": "ask",
                "selected_model": response["decision"]["recommended_model"],
                "executed": execute,
                "error": response["error"],
                "response_chars": len(response["output"]),
            })
            self._json(response, status=200 if not response.get("error") else 502)
            return

        if self.path == "/api/chat/stream":
            self._chat_stream(self._read_json())
            return
        if self.path == "/api/compare":
            self._compare(self._read_json())
            return

        if self.path == "/api/feedback":
            payload = self._read_json()
            write_event({
                "event": "feedback",
                "rating": payload.get("rating"),
                "selected_model": payload.get("model"),
                "profile": payload.get("profile"),
                "task_type": payload.get("task_type"),
            })
            self._json({"ok": True})
            return

        if self.path == "/api/models":
            payload = self._read_json()
            model_id = str(payload.get("id", ""))
            if not model_id:
                self._json({"error": "Model id is required"}, status=400)
                return
            settings = update_model(model_id, {"enabled": bool(payload.get("enabled"))})
            self._json({"ok": True, "settings": settings})
            return

        if self.path == "/api/settings":
            self._json({"ok": True, "settings": save_settings(self._read_json())})
            return

        self._json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _chat_stream(self, payload: dict[str, Any]) -> None:
        prompt = str(payload.get("prompt", ""))
        config = load_config(discover=True)
        decision = route(prompt, config, profile_name=payload.get("profile"))
        model = find_model(config, payload["model"]) if payload.get("model") else decision.selected.model
        session_id = str(payload.get("session_id", ""))
        if session_id:
            append_message(session_id, "user", prompt, {"profile": decision.profile_name})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._sse("route", {"decision": {**decision.as_dict(), "recommended_model": model["id"]}})
        output: list[str] = []
        try:
            for chunk in stream_with_metrics(model, prompt):
                output.append(chunk)
                self._sse("chunk", {"text": chunk})
            if session_id:
                append_message(session_id, "assistant", "".join(output), {"model": model["id"]})
            write_event({
                "event": "stream",
                "selected_model": model["id"],
                "response_chars": len("".join(output)),
                "profile": decision.profile_name,
            })
            self._sse("done", {"model": model["id"], "response_chars": len("".join(output))})
        except (BrokenPipeError, ConnectionResetError):
            return
        except ProviderError as exc:
            self._sse("error", {"error": str(exc)})

    def _v1_chat(self, payload: dict[str, Any]) -> None:
        messages = payload.get("messages", [])
        prompt = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in messages if isinstance(item, dict)
        )
        config = load_config(discover=True)
        decision = route(prompt, config, profile_name=payload.get("brainroute_profile"))
        requested = payload.get("model")
        model = find_model(config, requested) if requested and requested != "brainroute-auto" else decision.selected.model
        result = execute_with_fallback(model, decision, prompt)
        if result.error:
            self._json({"error": {"message": result.error, "type": "provider_error"}}, status=502)
            return
        self._json({
            "id": f"brainroute-{int(time.time())}",
            "object": "chat.completion",
            "model": result.model["id"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": result.output},
            }],
            "brainroute": {"decision": decision.as_dict(), "fallback_attempted": result.fallback_attempted},
        })

    def _compare(self, payload: dict[str, Any]) -> None:
        prompt = str(payload.get("prompt", ""))
        config = load_config(discover=True)
        decision = route(prompt, config, profile_name=payload.get("profile"))
        model_ids = payload.get("models") or [item.model["id"] for item in decision.ranked[:2]]
        comparisons = []
        for model_id in model_ids[:2]:
            model = find_model(config, model_id)
            item = {"model": model_id, "provider": model.get("provider"), "output": "", "error": None}
            if payload.get("execute"):
                result = execute_with_fallback(model, decision, prompt)
                item.update({
                    "model": result.model["id"],
                    "output": result.output,
                    "error": result.error,
                    "fallback_attempted": result.fallback_attempted,
                })
            comparisons.append(item)
        self._json({"decision": decision.as_dict(), "comparisons": comparisons})

    def _sse(self, event: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"))
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._security_headers()
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _static(self) -> None:
        requested = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        target = (STATIC_DIR / requested).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            target = STATIC_DIR / "index.html"
        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self._security_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _api_auth(self) -> bool:
        import os

        expected = os.environ.get("BRAINROUTE_API_KEY")
        if not expected:
            return True
        if self.headers.get("Authorization") == f"Bearer {expected}":
            return True
        self._json({"error": {"message": "Unauthorized", "type": "auth_error"}}, status=401)
        return False

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")


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
        "quality_score": model.get("quality_score"),
        "speed_score": model.get("speed_score"),
        "cost_score": model.get("cost_score"),
        "privacy_score": model.get("privacy_score"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainroute-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), BrainRouteHandler)
    print(f"BrainRoute running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
