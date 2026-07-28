from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any

from .classifier import TaskProfile, classify_prompt
from .config import RouterConfig
from .policy import PolicyResult, apply_policy
from .scorer import ScoredModel, effective_weights, score_models
from .settings import load_settings


@dataclass(frozen=True)
class RouteWarning:
    code: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class RouteDecision:
    task: TaskProfile
    selected: ScoredModel
    fallback: ScoredModel | None
    fallback_chain: tuple[ScoredModel, ...]
    ranked: tuple[ScoredModel, ...]
    profile_name: str
    policy: PolicyResult
    request_id: str
    exploration: bool = False
    warnings: tuple[RouteWarning, ...] = ()

    @property
    def margin(self) -> float:
        """Score gap to the runner-up. A small margin means a close call."""
        if len(self.ranked) < 2:
            return 1.0
        return round(self.ranked[0].score - self.ranked[1].score, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "profile": self.profile_name,
            "task_type": self.task.task_type,
            "complexity": self.task.complexity,
            "privacy_level": self.task.privacy_level,
            "urgency": self.task.urgency,
            "classifier": self.task.classifier,
            "classifier_detail": self.task.detail,
            "confidence": self.task.confidence,
            "capabilities": list(self.task.capabilities),
            "context_tokens": self.task.context_tokens,
            "recommended_model": self.selected.model["id"],
            "fallback_model": self.fallback.model["id"] if self.fallback else None,
            "fallback_chain": [item.model["id"] for item in self.fallback_chain],
            "score": self.selected.score,
            "margin": self.margin,
            "exploration": self.exploration,
            "local": bool(self.selected.model.get("local")),
            "warnings": [item.as_dict() for item in self.warnings],
            "reason": ", ".join(self.selected.reasons) or "highest weighted score",
            "weights": {
                key: round(value, 4)
                for key, value in (self.ranked[0].weights if self.ranked else {}).items()
            },
            "ranked": [
                {
                    "id": item.model["id"],
                    "provider": item.model.get("provider"),
                    "score": item.score,
                    "base_score": item.base_score,
                    "reasons": list(item.reasons),
                    "factors": {key: round(value, 4) for key, value in item.factors.items()},
                }
                for item in self.ranked
            ],
            "explain": [item.explain() for item in self.ranked],
            "policy": {
                "rejected": list(self.policy.rejected),
                "monthly_spend_usd": self.policy.monthly_spend_usd,
                "posture": self.policy.posture,
                "cloud_override_applied": self.policy.cloud_override_applied,
                "settings": self.policy.settings,
            },
        }


class NoRouteAvailable(ValueError):
    """Raised when policy and configuration leave no model to route to."""


def route(
    prompt: str,
    config: RouterConfig,
    profile_name: str | None = None,
    request_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> RouteDecision:
    settings = load_settings()
    resolved_profile_name = _resolve_profile(profile_name, config, settings)
    profile = config.profile(resolved_profile_name)
    task = classify_prompt(prompt, _classifier_model(config))
    policy = apply_policy(config.models, task, prompt, overrides=overrides)

    ranked = tuple(score_models(list(policy.allowed), profile, task))
    if not ranked:
        raise NoRouteAvailable(_no_route_detail(config, policy))

    ranked, exploring = _maybe_explore(ranked)
    chain = ranked[1:1 + _max_fallback_depth()]
    return RouteDecision(
        task=task,
        selected=ranked[0],
        fallback=chain[0] if chain else None,
        fallback_chain=chain,
        ranked=ranked,
        profile_name=resolved_profile_name,
        policy=policy,
        request_id=request_id or uuid.uuid4().hex[:16],
        exploration=exploring,
        warnings=_warnings(ranked, policy, settings),
    )


def _resolve_profile(
    profile_name: str | None, config: RouterConfig, settings: dict[str, Any]
) -> str:
    """An explicit profile always wins; otherwise the posture may set one."""
    if profile_name:
        return profile_name
    preferred = settings.get("policy", {}).get("posture_default_profile")
    if preferred and preferred in config.profiles:
        return str(preferred)
    return config.default_profile


def _warnings(
    ranked: tuple[ScoredModel, ...], policy: PolicyResult, settings: dict[str, Any]
) -> tuple[RouteWarning, ...]:
    warnings: list[RouteWarning] = []
    routing = settings.get("routing", {})

    if policy.cloud_override_applied:
        warnings.append(RouteWarning(
            "cloud_override",
            f"Cloud was enabled for this request only; the '{policy.posture}' posture blocks it by default.",
        ))

    if routing.get("warn_on_privacy_escalation", True):
        escalation = _privacy_escalation(ranked)
        if escalation:
            warnings.append(escalation)
    return tuple(warnings)


def _privacy_escalation(ranked: tuple[ScoredModel, ...]) -> RouteWarning | None:
    """Flag when the capability floor is what pushed a request off local hardware.

    A privacy trade should never be silent. This reports the counterfactual:
    the local model would have won if it had cleared the floor for this task.
    """
    selected = ranked[0]
    if selected.model.get("local"):
        return None

    local = next((item for item in ranked if item.model.get("local")), None)
    if local is None:
        return RouteWarning(
            "no_local_candidate",
            f"No local model was eligible, so this request goes to {selected.model['id']}.",
        )

    penalty = sum(item.value for item in local.adjustments if item.name == "capability_floor")
    if penalty >= 0 or local.score - penalty <= selected.score:
        return None
    return RouteWarning(
        "privacy_escalation",
        f"{local.model['id']} would have won ({local.score - penalty:.4f} vs {selected.score:.4f}) "
        f"but fell below the capability floor for this task, so it routes to "
        f"{selected.model['id']}. Lower routing.capability_floor.high to keep it local.",
    )


def preview_weights(config: RouterConfig, task: TaskProfile, profile_name: str | None = None) -> dict[str, float]:
    """Expose the effective weights for a task without running a full route."""
    return effective_weights(config.profile(profile_name), task)


def find_model(config: RouterConfig, model_id: str) -> dict[str, Any]:
    return config.model(model_id)


def _classifier_model(config: RouterConfig) -> dict[str, Any] | None:
    settings = load_settings()["classifier"]
    model_id = settings.get("model_id")
    if not (settings.get("enabled") and model_id):
        return None
    try:
        model = config.model(model_id)
    except ValueError:
        return None
    # Only local providers are used for classification: routing every request
    # through a paid API to decide where to send it defeats the purpose.
    return model if model.get("provider") == "ollama" else None


def _maybe_explore(ranked: tuple[ScoredModel, ...]) -> tuple[tuple[ScoredModel, ...], bool]:
    """Occasionally promote a near-tied candidate to gather evidence about it.

    A router that only ever picks the current best never learns whether its
    second choice would have been cheaper or faster. Exploration is off by
    default so routing stays deterministic unless an operator opts in.
    """
    routing = load_settings().get("routing", {})
    rate = float(routing.get("exploration_rate", 0.0) or 0.0)
    if rate <= 0 or len(ranked) < 2 or random.random() >= rate:
        return ranked, False

    margin = float(routing.get("exploration_score_margin", 0.05) or 0.05)
    best = ranked[0].score
    contenders = [index for index, item in enumerate(ranked[1:], start=1) if best - item.score <= margin]
    if not contenders:
        return ranked, False
    chosen = random.choice(contenders)
    reordered = (ranked[chosen], *[item for index, item in enumerate(ranked) if index != chosen])
    return reordered, True


def _max_fallback_depth() -> int:
    return max(1, int(load_settings().get("routing", {}).get("max_fallback_depth", 2)))


def _no_route_detail(config: RouterConfig, policy: PolicyResult) -> str:
    enabled = [model["id"] for model in config.models if model.get("enabled", True)]
    if not enabled:
        return "No enabled models are available. Enable a model in config/models.yaml or the UI."
    rejections = "; ".join(f"{item['id']}: {item['reason']}" for item in policy.rejected)
    return f"Every candidate was rejected by policy. {rejections}"
