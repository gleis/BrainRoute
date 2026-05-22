from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .classifier import TaskProfile
from .settings import load_settings
from .store import spend_since


@dataclass(frozen=True)
class PolicyResult:
    allowed: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, str], ...]
    settings: dict[str, Any]
    monthly_spend_usd: float


def apply_policy(models: list[dict[str, Any]], task: TaskProfile, prompt: str) -> PolicyResult:
    policy = load_settings()["policy"]
    monthly_spend = spend_since(datetime.now(timezone.utc).strftime("%Y-%m"))
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for model in models:
        reason = _reject_reason(model, task, prompt, policy, monthly_spend)
        if reason:
            rejected.append({"id": model["id"], "reason": reason})
            continue
        item = dict(model)
        if policy.get("prefer_local") and item.get("local"):
            item["policy_score_bonus"] = 0.2
        allowed.append(item)
    return PolicyResult(tuple(allowed), tuple(rejected), policy, monthly_spend)


def _reject_reason(
    model: dict[str, Any],
    task: TaskProfile,
    prompt: str,
    policy: dict[str, Any],
    monthly_spend: float,
) -> str | None:
    if not model.get("enabled", True):
        return "disabled"
    external = not model.get("local")
    if external and not policy.get("allow_cloud", True):
        return "cloud disabled by policy"
    if external and task.privacy_level == "high" and not policy.get("allow_cloud_for_private", False):
        return "private task cannot leave local providers"
    estimate = _estimate_cost(model, prompt)
    if estimate > float(policy.get("max_estimated_cost_usd", 0.25)):
        return "request budget exceeded"
    if external and monthly_spend >= float(policy.get("monthly_budget_usd", 50.0)):
        return "monthly budget reached"
    return None


def _estimate_cost(model: dict[str, Any], prompt: str) -> float:
    input_tokens = max(1, len(prompt) // 4) if prompt else 0
    return input_tokens * float(model.get("input_cost_per_token", 0))
