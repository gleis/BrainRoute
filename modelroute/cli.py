from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_config
from .evals import run_evals
from .execution import execute_with_fallback
from .postures import describe as describe_postures
from .providers import provider_health
from .reliability import snapshot
from .router import NoRouteAvailable, find_model, route
from .settings import clear_policy_overrides, load_settings, load_stored_settings, set_posture
from .store import prune
from .telemetry import read_events, write_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelroute")
    parser.add_argument("--discover", action="store_true", help="Include runtime-discovered models")
    subcommands = parser.add_subparsers(dest="command", required=True)

    route_parser = subcommands.add_parser("route", help="Rank models for a prompt")
    route_parser.add_argument("prompt")
    route_parser.add_argument("--profile")
    route_parser.add_argument("--json", action="store_true")
    route_parser.add_argument("--explain", action="store_true", help="Show the full score breakdown")
    route_parser.add_argument(
        "--allow-cloud", action="store_true",
        help="Opt in to cloud providers for this request only",
    )

    ask_parser = subcommands.add_parser("ask", help="Route a prompt and optionally execute it")
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--profile")
    ask_parser.add_argument("--model", help="Force a specific model id")
    ask_parser.add_argument("--execute", action="store_true")
    ask_parser.add_argument("--dry-run", action="store_true")
    ask_parser.add_argument(
        "--allow-cloud", action="store_true",
        help="Opt in to cloud providers for this request only",
    )

    posture_parser = subcommands.add_parser("posture", help="Show or set the privacy posture")
    posture_parser.add_argument("name", nargs="?", help="Posture to switch to")
    posture_parser.add_argument(
        "--reset-overrides", action="store_true",
        help="Drop pinned policy flags so the posture governs them again",
    )

    subcommands.add_parser("policy", help="Show the resolved routing policy")
    subcommands.add_parser("models", help="List configured models")
    subcommands.add_parser("health", help="Check configured provider availability")
    subcommands.add_parser("reliability", help="Show observed per-model reliability")
    subcommands.add_parser("prune", help="Trim telemetry tables to their retention cap")

    telemetry_parser = subcommands.add_parser("telemetry", help="Show recent telemetry events")
    telemetry_parser.add_argument("--limit", type=int, default=10)

    eval_parser = subcommands.add_parser("eval", help="Run routing evals")
    eval_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    config = load_config(discover=args.discover)
    handlers = {
        "route": _cmd_route,
        "ask": _cmd_ask,
        "posture": _cmd_posture,
        "policy": _cmd_policy,
        "models": _cmd_models,
        "health": _cmd_health,
        "reliability": _cmd_reliability,
        "prune": _cmd_prune,
        "telemetry": _cmd_telemetry,
        "eval": _cmd_eval,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return 1
    try:
        return handler(args, config)
    except NoRouteAvailable as exc:
        print(f"No route available: {exc}", file=sys.stderr)
        return 3


def _cmd_route(args: argparse.Namespace, config: Any) -> int:
    decision = route(
        args.prompt, config, profile_name=args.profile, overrides=_overrides(args)
    )
    payload = decision.as_dict()
    write_event({
        "event": "route",
        "request_id": decision.request_id,
        "selected_model": payload["recommended_model"],
        "decision": payload,
        "profile": decision.profile_name,
    })
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_decision(payload)
        if args.explain:
            _print_explanation(payload)
    return 0


def _cmd_ask(args: argparse.Namespace, config: Any) -> int:
    decision = route(
        args.prompt, config, profile_name=args.profile, overrides=_overrides(args)
    )
    model = find_model(config, args.model) if args.model else decision.selected.model
    dry_run = not args.execute or args.dry_run
    event = {
        "event": "ask",
        "request_id": decision.request_id,
        "profile": decision.profile_name,
        "selected_model": model["id"],
        "router_model": decision.selected.model["id"],
        "dry_run": dry_run,
    }
    if dry_run:
        write_event(event)
        _print_decision({**decision.as_dict(), "recommended_model": model["id"]})
        print("\nDry run only. Add --execute to call the provider.")
        return 0

    result = execute_with_fallback(model, decision, args.prompt)
    if result.error:
        write_event({**event, "error": result.error})
        print(f"Provider call failed: {result.error}", file=sys.stderr)
        return 2
    if result.fallback_attempted:
        print(f"[fell back to {result.model['id']}]", file=sys.stderr)
    write_event({**event, "selected_model": result.model["id"], "response_chars": len(result.output)})
    print(result.output)
    return 0


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {"allow_cloud": True} if getattr(args, "allow_cloud", False) else {}


def _cmd_posture(args: argparse.Namespace, config: Any) -> int:
    if args.reset_overrides:
        clear_policy_overrides()
        print("Pinned policy flags cleared; the posture now governs them.")
    if args.name:
        if args.name not in {item["id"] for item in describe_postures()}:
            available = ", ".join(item["id"] for item in describe_postures())
            print(f"Unknown posture '{args.name}'. Available: {available}", file=sys.stderr)
            return 2
        set_posture(args.name)
        print(f"Posture set to {args.name}.")

    active = load_settings()["policy"]["posture"]
    print(f"\nActive posture: {active}\n")
    for item in describe_postures():
        marker = "*" if item["id"] == active else " "
        print(f" {marker} {item['id']:<13} {item['label']}")
        print(f"   {' ' * 13} {item['description']}")
    return 0


def _cmd_policy(args: argparse.Namespace, config: Any) -> int:
    settings = load_settings()
    policy = settings["policy"]
    stored = load_stored_settings().get("policy", {})
    print(f"Posture:              {policy['posture']}")
    print(f"Default profile:      {policy.get('posture_default_profile') or config.default_profile}")
    print(f"Per-request override: {'allowed' if policy['cloud_override_allowed'] else 'blocked'}")
    print("\nResolved policy (* = pinned by you, overriding the posture):")
    for key in sorted(policy):
        if key in {"posture", "cloud_override_allowed", "posture_default_profile"}:
            continue
        marker = "*" if key in stored else " "
        print(f" {marker} {key:<26} {policy[key]}")

    floors = settings["routing"]["capability_floor"]
    print("\nCapability floors (minimum quality by task complexity):")
    for tier in ("low", "medium", "high"):
        print(f"   {tier:<26} {floors.get(tier)}")
    print(f"   {'penalty per point':<26} {settings['routing']['capability_floor_penalty']}")
    return 0


def _cmd_models(args: argparse.Namespace, config: Any) -> int:
    header = f"{'ID':<26} {'PROVIDER':<12} {'WHERE':<6} {'STATE':<9} {'$/MTOK':>9}  NAME"
    print(header)
    print("-" * len(header))
    for model in sorted(config.models, key=lambda item: item["id"]):
        enabled = "enabled" if model.get("enabled", True) else "disabled"
        local = "local" if model.get("local") else "cloud"
        price = model.get("input_price_per_mtok")
        price_text = f"{price:.2f}" if isinstance(price, (int, float)) else "?"
        print(
            f"{model['id']:<26} {str(model.get('provider')):<12} {local:<6} "
            f"{enabled:<9} {price_text:>9}  {model.get('name')}"
        )
    return 0


def _cmd_health(args: argparse.Namespace, config: Any) -> int:
    for item in provider_health(config.models):
        status = "ok" if item["ok"] else "not ok"
        circuit = item.get("circuit", "-")
        print(f"{item['id']:<26} {status:<7} {circuit:<10} {item['detail']}")
    return 0


def _cmd_reliability(args: argparse.Namespace, config: Any) -> int:
    entries = snapshot()
    if not entries:
        print("No telemetry recorded yet.")
        return 0
    print(f"{'ID':<26} {'CIRCUIT':<11} {'RUNS':>6} {'OK':>7} {'P95 MS':>8}")
    for entry in sorted(entries.values(), key=lambda item: item.model_id):
        print(
            f"{entry.model_id:<26} {entry.state:<11} {entry.samples:>6} "
            f"{entry.ok_rate:>6.0%} {entry.p95_latency_ms:>8}"
        )
    return 0


def _cmd_prune(args: argparse.Namespace, config: Any) -> int:
    result = prune()
    print(f"Deleted {result['events_deleted']} events and {result['runs_deleted']} runs.")
    return 0


def _cmd_telemetry(args: argparse.Namespace, config: Any) -> int:
    for event in read_events(limit=args.limit):
        label = event.get("selected_model") or event.get("decision", {}).get("recommended_model") or ""
        print(f"{event.get('created_at', '')} {event.get('event', 'event'):<10} {label}")
    return 0


def _cmd_eval(args: argparse.Namespace, config: Any) -> int:
    results = run_evals(config)
    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2))
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name} (margin {result.margin:+.3f})")
            for failure in result.failures:
                print(f"     {failure}")
    passed = sum(1 for result in results if result.passed)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def _print_decision(decision: dict[str, Any]) -> None:
    where = "local" if decision.get("local") else "cloud"
    print(f"Recommended: {decision['recommended_model']} ({decision['score']:.4f}) [{where}]")
    if decision.get("fallback_chain"):
        print(f"Fallbacks:   {' -> '.join(decision['fallback_chain'])}")
    print(f"Task:        {decision['task_type']} / {decision['complexity']} "
          f"(confidence {decision['confidence']:.2f}, via {decision['classifier']})")
    print(f"Privacy:     {decision['privacy_level']}")
    print(f"Urgency:     {decision['urgency']}")
    print(f"Margin:      {decision['margin']:+.4f}")
    print(f"Reason:      {decision['reason']}")

    weights = decision.get("weights") or {}
    if weights:
        print("Weights:     " + "  ".join(f"{key}={value:.2f}" for key, value in weights.items()))

    print("\nRanking:")
    for item in decision["ranked"]:
        print(f"  {item['score']:.4f}  {item['id']:<26} [{item['provider']}]")

    for warning in decision.get("warnings", []):
        print(f"\n! {warning['code']}: {warning['message']}")

    rejected = decision.get("policy", {}).get("rejected", [])
    if rejected:
        print("\nRejected by policy:")
        for item in rejected:
            print(f"  {item['id']:<26} {item['reason']}")


def _print_explanation(decision: dict[str, Any]) -> None:
    print("\nScore breakdown:")
    for item in decision.get("explain", []):
        print(f"\n  {item['id']}  final={item['score']:.4f}  base={item['base_score']:.4f}")
        for factor, value in item["contributions"].items():
            print(f"    {factor:<10} {item['factors'][factor]:.3f} x {item['weights'][factor]:.3f} = {value:+.4f}")
        for adjustment in item["adjustments"]:
            print(f"    {adjustment['name']:<20} {adjustment['value']:+.4f}  {adjustment['detail']}")


if __name__ == "__main__":
    raise SystemExit(main())
