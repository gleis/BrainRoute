from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .evals import run_evals
from .providers import generate, provider_health
from .router import find_model, route
from .telemetry import read_events, write_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainpicker")
    subcommands = parser.add_subparsers(dest="command", required=True)

    route_parser = subcommands.add_parser("route", help="Rank models for a prompt")
    route_parser.add_argument("prompt")
    route_parser.add_argument("--profile")
    route_parser.add_argument("--json", action="store_true")

    ask_parser = subcommands.add_parser("ask", help="Route a prompt and optionally execute it")
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--profile")
    ask_parser.add_argument("--model", help="Force a specific model id")
    ask_parser.add_argument("--execute", action="store_true")
    ask_parser.add_argument("--dry-run", action="store_true")

    subcommands.add_parser("models", help="List configured models")
    subcommands.add_parser("health", help="Check configured provider availability")

    telemetry_parser = subcommands.add_parser("telemetry", help="Show recent telemetry events")
    telemetry_parser.add_argument("--limit", type=int, default=10)

    eval_parser = subcommands.add_parser("eval", help="Run routing evals")
    eval_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    config = load_config()

    if args.command == "route":
        decision = route(args.prompt, config, profile_name=args.profile)
        write_event({"event": "route", "decision": decision.as_dict(), "profile": decision.profile_name})
        if args.json:
            print(json.dumps(decision.as_dict(), indent=2))
        else:
            _print_decision(decision.as_dict())
        return 0

    if args.command == "ask":
        decision = route(args.prompt, config, profile_name=args.profile)
        model = find_model(config, args.model) if args.model else decision.selected.model
        event = {
            "event": "ask",
            "profile": decision.profile_name,
            "selected_model": model["id"],
            "router_model": decision.selected.model["id"],
            "dry_run": not args.execute or args.dry_run,
        }
        if not args.execute or args.dry_run:
            write_event(event)
            _print_decision({**decision.as_dict(), "recommended_model": model["id"]})
            print("\nDry run only. Add --execute to call the provider.")
            return 0

        try:
            output = generate(model, args.prompt)
        except Exception as exc:  # Provider errors should be readable in CLI output.
            write_event({**event, "error": str(exc)})
            print(f"Provider call failed: {exc}", file=sys.stderr)
            return 2
        write_event({**event, "response_chars": len(output)})
        print(output)
        return 0

    if args.command == "models":
        for model in config.models:
            enabled = "enabled" if model.get("enabled", True) else "disabled"
            local = "local" if model.get("local") else "cloud"
            print(f"{model['id']:<18} {model.get('provider'):<8} {local:<5} {enabled:<8} {model.get('name')}")
        return 0

    if args.command == "health":
        for item in provider_health(config.models):
            status = "ok" if item["ok"] else "not ok"
            print(f"{item['id']:<18} {status:<6} {item['detail']}")
        return 0

    if args.command == "telemetry":
        for event in read_events(limit=args.limit):
            label = event.get("selected_model") or event.get("decision", {}).get("recommended_model") or ""
            print(f"{event.get('created_at', '')} {event.get('event', 'event'):<8} {label}")
        return 0

    if args.command == "eval":
        results = run_evals(config)
        if args.json:
            print(json.dumps([result.__dict__ for result in results], indent=2))
        else:
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                print(f"{status} {result.name}: expected {result.expected_model}, got {result.actual_model}")
        return 0 if all(result.passed for result in results) else 1

    return 1


def _print_decision(decision: dict) -> None:
    print(f"Recommended: {decision['recommended_model']} ({decision['score']})")
    if decision.get("fallback_model"):
        print(f"Fallback:    {decision['fallback_model']}")
    print(f"Task:        {decision['task_type']} / {decision['complexity']}")
    print(f"Privacy:     {decision['privacy_level']}")
    print(f"Urgency:     {decision['urgency']}")
    print(f"Reason:      {decision['reason']}")
    print("\nRanking:")
    for item in decision["ranked"]:
        print(f"  {item['score']:.4f}  {item['id']}  [{item['provider']}]")


if __name__ == "__main__":
    raise SystemExit(main())
