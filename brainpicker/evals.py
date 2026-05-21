from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ROOT, RouterConfig
from .router import route
from .simple_yaml import load_simple_yaml


DEFAULT_EVALS_PATH = ROOT / "config" / "evals.yaml"


@dataclass(frozen=True)
class EvalResult:
    name: str
    expected_model: str
    actual_model: str
    passed: bool
    profile: str


def run_evals(config: RouterConfig, path: str | Path = DEFAULT_EVALS_PATH) -> list[EvalResult]:
    data = load_simple_yaml(path)
    results: list[EvalResult] = []
    for case in data.get("cases", []):
        decision = route(case["prompt"], config, profile_name=case.get("profile"))
        actual = decision.selected.model["id"]
        expected = case["expected_model"]
        results.append(EvalResult(
            name=case["name"],
            expected_model=expected,
            actual_model=actual,
            passed=actual == expected,
            profile=case.get("profile", config.default_profile),
        ))
    return results

