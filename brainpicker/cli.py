from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .providers import generate
from .router import find_model, route
from .telemetry import write_event


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

