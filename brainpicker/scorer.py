from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classifier import TaskProfile


@dataclass(frozen=True)
class ScoredModel:
    model: dict[str, Any]
    score: float
    reasons: tuple[str, ...]


def score_models(
    models: list[dict[str, Any]],
    profile: dict[str, float],
    task: TaskProfile,
) -> list[ScoredModel]:
    scored = [
        _score_model(model, profile, task)
        for model in models
        if model.get("enabled", True)
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)


def _score_model(model: dict[str, Any], profile: dict[str, float], task: TaskProfile) -> ScoredModel:
    reasons: list[str] = []
    base = (
        float(model.get("quality_score", 0)) * profile.get("quality", 0)
        + float(model.get("speed_score", 0)) * profile.get("speed", 0)
        + float(model.get("cost_score", 0)) * profile.get("cost", 0)
        + float(model.get("privacy_score", 0)) * profile.get("privacy", 0)
    )

    strengths = set(model.get("strengths", []))
    weaknesses = set(model.get("weaknesses", []))
    task_fit = 0.0
    if task.task_type in strengths:
        task_fit += profile.get("task_fit", 0)
        reasons.append(f"matches {task.task_type}")
    if task.task_type in weaknesses:
        task_fit -= profile.get("task_fit", 0)
        reasons.append(f"weak for {task.task_type}")
    if task.privacy_level == "high" and model.get("local"):
        task_fit += 0.15
        reasons.append("keeps sensitive prompt local")
    if task.privacy_level == "high" and not model.get("local"):
        task_fit -= 0.20
        reasons.append("external provider privacy penalty")
    if task.urgency == "high":
        task_fit += float(model.get("speed_score", 0)) * 0.05
        reasons.append("urgency favors speed")
    if task.complexity == "high":
        task_fit += float(model.get("quality_score", 0)) * 0.05
        reasons.append("complex task favors quality")
        if float(model.get("quality_score", 0)) >= 0.9:
            task_fit += 0.25
            reasons.append("premium quality bonus")

    return ScoredModel(model=model, score=round(base + task_fit, 4), reasons=tuple(reasons))
