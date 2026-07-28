"""Hard admission rules applied before any model is scored.

Scoring expresses preferences; policy expresses constraints. A model rejected
here is not a worse choice, it is not a legal choice: it cannot hold the
prompt, it would breach a budget, it would leak a sensitive request, or its
provider is currently failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .classifier import TaskProfile
from .pricing import ASSUMED_OUTPUT_TOKENS, estimate_request_cost
from .reliability import OPEN, snapshot
from .settings import load_settings
from .store import spend_since


UNHEALTHY_PREFIX = "provider unhealthy"


@dataclass(frozen=True)
class PolicyResult:
    allowed: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, str], ...]
    settings: dict[str, Any]
    monthly_spend_usd: float
    posture: str = ""
    cloud_override_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": [model["id"] for model in self.allowed],
            "rejected": list(self.rejected),
            "monthly_spend_usd": self.monthly_spend_usd,
            "posture": self.posture,
            "cloud_override_applied": self.cloud_override_applied,
            "settings": self.settings,
        }


def apply_policy(
    models: list[dict[str, Any]],
    task: TaskProfile,
    prompt: str,
    overrides: dict[str, Any] | None = None,
) -> PolicyResult:
    policy = dict(load_settings()["policy"])
    posture = str(policy.get("posture", ""))
    cloud_override = _apply_cloud_override(policy, overrides)
    monthly_spend = spend_since(datetime.now(timezone.utc).strftime("%Y-%m"))
    health = snapshot()
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for model in models:
        reason = _reject_reason(model, task, prompt, policy, monthly_spend, health)
        if reason:
            rejected.append({"id": model["id"], "reason": reason})
            continue
        item = dict(model)
        if policy.get("prefer_local") and item.get("local"):
            item["policy_score_bonus"] = float(policy.get("local_bonus", 0.15))
        allowed.append(item)

    # Never strand a request because every provider tripped its breaker. If the
    # only rejections were reliability-based, re-admit those candidates and let
    # scoring penalise them instead of failing the request outright.
    if not allowed:
        unhealthy_ids = {
            item["id"] for item in rejected if item["reason"].startswith(UNHEALTHY_PREFIX)
        }
        recovered = [dict(model) for model in models if model["id"] in unhealthy_ids]
        if recovered:
            allowed = recovered
            rejected = [
                item for item in rejected if not item["reason"].startswith(UNHEALTHY_PREFIX)
            ]

    return PolicyResult(
        allowed=tuple(allowed),
        rejected=tuple(rejected),
        settings=policy,
        monthly_spend_usd=monthly_spend,
        posture=posture,
        cloud_override_applied=cloud_override,
    )


def _apply_cloud_override(policy: dict[str, Any], overrides: dict[str, Any] | None) -> bool:
    """Honour a per-request opt-in to cloud, where the posture permits one.

    This only lifts the blanket ``allow_cloud`` block. It deliberately does not
    touch ``allow_cloud_for_private``: opting in to cloud for a request is a
    different decision from consenting to send sensitive data off the machine,
    and conflating them is how privacy guarantees get lost.
    """
    if not overrides or not overrides.get("allow_cloud"):
        return False
    if not policy.get("cloud_override_allowed", True):
        return False
    if policy.get("allow_cloud"):
        return False
    policy["allow_cloud"] = True
    return True


def _reject_reason(
    model: dict[str, Any],
    task: TaskProfile,
    prompt: str,
    policy: dict[str, Any],
    monthly_spend: float,
    health: dict[str, Any],
) -> str | None:
    if not model.get("enabled", True):
        return "disabled"

    external = not model.get("local")
    if external and not policy.get("allow_cloud", True):
        return "cloud disabled by policy"
    if external and task.privacy_level == "high" and not policy.get("allow_cloud_for_private", False):
        return "private task cannot leave local providers"

    context_reason = _context_reason(model, task, policy)
    if context_reason:
        return context_reason

    estimate = estimate_request_cost(model, task.context_tokens or _tokens(prompt))
    if estimate > float(policy.get("max_estimated_cost_usd", 0.25)):
        return f"estimated ${estimate:.4f} exceeds the per-request budget"
    if external and monthly_spend >= float(policy.get("monthly_budget_usd", 50.0)):
        return f"monthly budget reached (${monthly_spend:.2f})"

    entry = health.get(model["id"])
    if entry is not None and entry.state == OPEN:
        return f"{UNHEALTHY_PREFIX}: {entry.detail}"
    return None


def _context_reason(model: dict[str, Any], task: TaskProfile, policy: dict[str, Any]) -> str | None:
    """Reject models that physically cannot hold the request.

    Without this the router would happily send a 200k-token prompt to a model
    with a 32k window and only discover the problem as a provider error.
    """
    max_context = int(model.get("max_context", 0) or 0)
    if not max_context:
        return None
    headroom = int(policy.get("context_headroom_tokens", 512))
    required = task.context_tokens + ASSUMED_OUTPUT_TOKENS + headroom
    if required > max_context:
        return f"context window too small ({max_context} tokens, needs ~{required})"
    return None


def _tokens(prompt: str) -> int:
    return max(1, len(prompt) // 4) if prompt else 0
