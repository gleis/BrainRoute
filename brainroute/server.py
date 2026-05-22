from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .catalog import discover_models
from .config import load_config
from .providers import ProviderError, generate, provider_health
from .router import find_model, route
from .settings import load_settings, save_settings, update_model
from .telemetry import read_events, write_event


STATIC_DIR = Path(__file__).resolve().parent / "static"


class BrainRouteHandler(BaseHTTPRequestHandler):
    server_version = "BrainRoute/0.1"

    def do_GET(self) -> None:
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
        self._static()

    def do_POST(self) -> None:
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
                try:
                    response["output"] = generate(model, prompt)
                except ProviderError as exc:
                    response["error"] = str(exc)
                    fallback = decision.fallback.model if decision.fallback else None
                    if fallback and fallback["id"] != model["id"]:
                        response["fallback_attempted"] = fallback["id"]
                        try:
                            response["output"] = generate(fallback, prompt)
                            response["error"] = None
                            response["decision"]["recommended_model"] = fallback["id"]
                        except ProviderError as fallback_exc:
                            response["fallback_error"] = str(fallback_exc)
            write_event({
                "event": "ask",
                "selected_model": response["decision"]["recommended_model"],
                "executed": execute,
                "error": response["error"],
                "response_chars": len(response["output"]),
            })
            self._json(response, status=200 if not response.get("error") else 502)
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
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
