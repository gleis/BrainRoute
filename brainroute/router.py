from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classifier import TaskProfile, classify, classify_with_model
from .config import RouterConfig
from .scorer import ScoredModel, score_models
from .settings import load_settings


@dataclass(frozen=True)
class RouteDecision:
    task: TaskProfile
    selected: ScoredModel
    fallback: ScoredModel | None
    ranked: tuple[ScoredModel, ...]
    profile_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
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
            "score": self.selected.score,
            "reason": ", ".join(self.selected.reasons) or "highest weighted score",
            "ranked": [
                {
                    "id": item.model["id"],
                    "provider": item.model.get("provider"),
                    "score": item.score,
                    "reasons": list(item.reasons),
                }
                for item in self.ranked
            ],
        }


def route(prompt: str, config: RouterConfig, profile_name: str | None = None) -> RouteDecision:
    resolved_profile_name = profile_name or config.default_profile
    task = _classify(prompt, config)
    ranked = tuple(score_models(config.models, config.profile(resolved_profile_name), task))
    if not ranked:
        raise ValueError("No enabled models are available")
    fallback = ranked[1] if len(ranked) > 1 else None
    return RouteDecision(
        task=task,
        selected=ranked[0],
        fallback=fallback,
        ranked=ranked,
        profile_name=resolved_profile_name,
    )


def _classify(prompt: str, config: RouterConfig) -> TaskProfile:
    settings = load_settings()["classifier"]
    model_id = settings.get("model_id")
    if settings.get("enabled") and model_id:
        try:
            model = find_model(config, model_id)
            if model.get("provider") == "ollama":
                return classify_with_model(prompt, model)
        except Exception as exc:
            heuristic = classify(prompt)
            return TaskProfile(
                **{
                    **heuristic.__dict__,
                    "detail": f"router model unavailable: {exc}"[:240],
                }
            )
    return classify(prompt)


def find_model(config: RouterConfig, model_id: str) -> dict[str, Any]:
    for model in config.models:
        if model.get("id") == model_id:
            return model
    raise ValueError(f"Unknown model '{model_id}'")
