"""Routing evals.

These assert routing behaviour without calling any provider, so they are cheap
enough to run on every weight change. Beyond "did we pick the expected model",
a case can assert the classification itself, which catches the common failure
where the router picks the right model for the wrong reason.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RouterConfig
from .config_paths import CONFIG_DIR
from .router import NoRouteAvailable, route
from .settings import override
from .simple_yaml import load_simple_yaml


DEFAULT_EVALS_PATH = CONFIG_DIR / "evals.yaml"


@dataclass(frozen=True)
class EvalResult:
    name: str
    expected_model: str
    actual_model: str
    passed: bool
    profile: str
    posture: str = ""
    margin: float = 0.0
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_model": self.expected_model,
            "actual_model": self.actual_model,
            "passed": self.passed,
            "profile": self.profile,
            "posture": self.posture,
            "margin": self.margin,
            "warnings": list(self.warnings),
            "failures": list(self.failures),
        }


def run_evals(config: RouterConfig, path: str | Path = DEFAULT_EVALS_PATH) -> list[EvalResult]:
    data = load_simple_yaml(path)
    return [_run_case(case, config) for case in data.get("cases", [])]


def _run_case(case: dict[str, Any], config: RouterConfig) -> EvalResult:
    # A case may pin a posture so behaviour under each privacy stance is
    # asserted explicitly rather than depending on the operator's current one.
    posture = case.get("posture")
    scope = override({"policy": {"posture": posture}}) if posture else nullcontext()
    with scope:
        return _evaluate(case, config)


def _evaluate(case: dict[str, Any], config: RouterConfig) -> EvalResult:
    profile = case.get("profile") or config.default_profile
    expected = str(case.get("expected_model", ""))
    try:
        decision = route(case["prompt"], config, profile_name=case.get("profile"))
    except NoRouteAvailable as exc:
        return EvalResult(
            name=case["name"],
            expected_model=expected,
            actual_model="",
            passed=False,
            profile=profile,
            failures=(f"no route available: {exc}",),
        )

    actual = decision.selected.model["id"]
    failures: list[str] = []
    if expected and actual != expected:
        failures.append(f"expected model {expected}, got {actual}")

    for field, attribute in (
        ("expected_task_type", "task_type"),
        ("expected_complexity", "complexity"),
        ("expected_privacy", "privacy_level"),
        ("expected_urgency", "urgency"),
    ):
        wanted = case.get(field)
        if wanted and str(getattr(decision.task, attribute)) != str(wanted):
            failures.append(f"{attribute}: expected {wanted}, got {getattr(decision.task, attribute)}")

    min_margin = case.get("min_margin")
    if min_margin is not None and decision.margin < float(min_margin):
        failures.append(f"margin {decision.margin} below required {min_margin}")

    codes = {item.code for item in decision.warnings}
    wanted_warning = case.get("expected_warning")
    if wanted_warning and wanted_warning not in codes:
        failures.append(f"expected warning {wanted_warning}, got {sorted(codes) or 'none'}")
    if case.get("expect_no_warnings") and codes:
        failures.append(f"expected no warnings, got {sorted(codes)}")

    return EvalResult(
        name=case["name"],
        expected_model=expected,
        actual_model=actual,
        passed=not failures,
        profile=profile,
        posture=str(case.get("posture") or ""),
        margin=decision.margin,
        warnings=tuple(sorted(codes)),
        failures=tuple(failures),
    )
