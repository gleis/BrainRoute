"""Model scoring.

The scoring model has three layers, applied in order:

1. **Effective factor values** - declared scores from the catalog, blended with
   what telemetry actually observed for that model.
2. **Task-modulated weights** - the routing profile sets the operator's
   priorities; the classified task then tilts those weights. A hard task shifts
   weight toward quality and task fit; an urgent one toward speed. Weights are
   renormalised after tilting so scores stay comparable across profiles.
3. **Adjustments** - a small, named, bounded set of additive corrections for
   things that are not smooth trade-offs: capability floors, privacy handling,
   degraded providers, and context pressure.

Every step is recorded on the result so a decision can be explained after the
fact rather than inferred from a single opaque number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .classifier import TaskProfile
from .config import FACTORS, normalize_profile
from .reliability import HALF_OPEN, ModelReliability, blend, observed_speed_score, snapshot
from .settings import load_settings


# Minimum credible quality per complexity tier. A model below the floor is not
# a plausible choice for the task no matter how cheap or fast it is; this is
# what stops a small local model winning "design a production architecture"
# purely on cost and privacy.
#
# These are defaults. Both the floors and the penalty are configurable under
# ``routing`` in settings - lowering the "high" floor keeps a smaller local
# model in contention on hard tasks, trading answer quality for privacy.
CAPABILITY_FLOOR = {"low": 0.0, "medium": 0.55, "high": 0.80}
FLOOR_PENALTY_PER_POINT = 1.5
HEADROOM_BONUS_PER_POINT = 0.4
MAX_HEADROOM_BONUS = 0.10

# Multiplicative weight tilts applied when a task signal is present. Applied
# before renormalisation, so these express relative emphasis, not absolute size.
COMPLEXITY_TILTS = {
    "high": {"quality": 1.8, "task_fit": 1.5, "cost": 0.5, "speed": 0.6},
    "medium": {},
    "low": {"quality": 0.7, "cost": 1.3, "speed": 1.2},
}
URGENCY_TILT = {"speed": 1.6, "cost": 0.8}
PRIVACY_TILT = {"privacy": 2.0}

# Fraction of a model's context window above which we start to distrust it.
CONTEXT_PRESSURE_RATIO = 0.6


@dataclass(frozen=True)
class Adjustment:
    name: str
    value: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": round(self.value, 4), "detail": self.detail}


@dataclass(frozen=True)
class ScoredModel:
    model: dict[str, Any]
    score: float
    base_score: float
    factors: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    adjustments: tuple[Adjustment, ...] = ()
    reliability: ModelReliability | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        """Human-readable explanation, ordered by absolute impact."""
        ranked = sorted(self.adjustments, key=lambda item: abs(item.value), reverse=True)
        return tuple(item.detail for item in ranked if abs(item.value) >= 0.005)

    def explain(self) -> dict[str, Any]:
        return {
            "id": self.model.get("id"),
            "score": self.score,
            "base_score": self.base_score,
            "factors": {key: round(value, 4) for key, value in self.factors.items()},
            "weights": {key: round(value, 4) for key, value in self.weights.items()},
            "contributions": {key: round(value, 4) for key, value in self.contributions.items()},
            "adjustments": [item.as_dict() for item in self.adjustments],
            "reliability": self.reliability.as_dict() if self.reliability else None,
        }


def effective_weights(profile: dict[str, float], task: TaskProfile) -> dict[str, float]:
    """Normalised profile weights, tilted by the classified task, renormalised."""
    weights = dict(normalize_profile(profile))
    tilts: dict[str, float] = {}
    for source in (
        COMPLEXITY_TILTS.get(task.complexity, {}),
        URGENCY_TILT if task.urgency == "high" else {},
        PRIVACY_TILT if task.privacy_level == "high" else {},
    ):
        for factor, multiplier in source.items():
            tilts[factor] = tilts.get(factor, 1.0) * multiplier

    tilted = {factor: weights[factor] * tilts.get(factor, 1.0) for factor in FACTORS}
    total = sum(tilted.values())
    if total <= 0:
        return weights
    return {factor: value / total for factor, value in tilted.items()}


def capability_tuning() -> dict[str, Any]:
    """Configured capability floors and penalty, falling back to defaults."""
    routing = load_settings().get("routing", {})
    configured = routing.get("capability_floor") or {}
    floors = {
        tier: _clamp(float(configured.get(tier, default)))
        for tier, default in CAPABILITY_FLOOR.items()
    }
    try:
        penalty = max(0.0, float(routing.get("capability_floor_penalty", FLOOR_PENALTY_PER_POINT)))
    except (TypeError, ValueError):
        penalty = FLOOR_PENALTY_PER_POINT
    return {"floors": floors, "penalty": penalty}


def score_models(
    models: list[dict[str, Any]],
    profile: dict[str, float],
    task: TaskProfile,
) -> list[ScoredModel]:
    weights = effective_weights(profile, task)
    health = snapshot()
    tuning = capability_tuning()
    scored = [
        _score_model(
            model, weights, task, health.get(model["id"]) or ModelReliability(model["id"]), tuning
        )
        for model in models
        if model.get("enabled", True)
    ]
    # Ties are broken by model id so routing is deterministic and reproducible.
    return sorted(scored, key=lambda item: (-item.score, item.model.get("id", "")))


def _score_model(
    model: dict[str, Any],
    weights: dict[str, float],
    task: TaskProfile,
    health: ModelReliability,
    tuning: dict[str, Any],
) -> ScoredModel:
    factors = {
        "quality": _quality_factor(model, health),
        "speed": _speed_factor(model, health),
        "cost": _clamp(float(model.get("cost_score", 0.5))),
        "privacy": _clamp(float(model.get("privacy_score", 0.5))),
        "task_fit": _task_fit_factor(model, task),
    }
    contributions = {factor: factors[factor] * weights.get(factor, 0.0) for factor in FACTORS}
    base = sum(contributions.values())
    adjustments = _adjustments(model, task, factors, health, tuning)
    total = base + sum(item.value for item in adjustments)
    return ScoredModel(
        model=model,
        score=round(_clamp(total), 4),
        base_score=round(base, 4),
        factors=factors,
        weights=weights,
        contributions=contributions,
        adjustments=adjustments,
        reliability=health,
    )


def _quality_factor(model: dict[str, Any], health: ModelReliability) -> float:
    """Declared quality, corrected by human feedback once enough votes exist."""
    declared = _clamp(float(model.get("quality_score", 0.5)))
    if health.feedback_score is None or health.feedback_votes < 3:
        return declared
    confidence = min(1.0, health.feedback_votes / 20)
    return _clamp(blend(declared, health.feedback_score, confidence))


def _speed_factor(model: dict[str, Any], health: ModelReliability) -> float:
    """Declared speed, replaced by measured latency as samples accumulate."""
    declared = _clamp(float(model.get("speed_score", 0.5)))
    if not health.p95_latency_ms:
        return declared
    return _clamp(blend(declared, observed_speed_score(health.p95_latency_ms), health.confidence))


def _task_fit_factor(model: dict[str, Any], task: TaskProfile) -> float:
    """How well the model's declared aptitudes match the classified task.

    Centred on 0.5 (no information) so that a model with no declared strengths
    is neither rewarded nor punished.
    """
    strengths = {str(item).lower() for item in model.get("strengths", [])}
    weaknesses = {str(item).lower() for item in model.get("weaknesses", [])}
    task_type = task.task_type.lower()

    fit = 0.5
    if task_type in strengths:
        fit += 0.35
    if task_type in weaknesses:
        fit -= 0.35
    if task_type not in strengths and "general" in strengths:
        fit += 0.05

    capabilities = {str(item).lower() for item in task.capabilities}
    if capabilities:
        overlap = len(capabilities & strengths) / len(capabilities)
        fit += 0.15 * overlap
        if capabilities & weaknesses:
            fit -= 0.10

    if task.context_tokens > 60_000 and "long_context" in strengths:
        fit += 0.10
    return _clamp(fit)


def _adjustments(
    model: dict[str, Any],
    task: TaskProfile,
    factors: dict[str, float],
    health: ModelReliability,
    tuning: dict[str, Any],
) -> tuple[Adjustment, ...]:
    adjustments: list[Adjustment] = []
    quality = factors["quality"]

    floor = tuning["floors"].get(task.complexity, 0.0)
    if quality < floor:
        shortfall = floor - quality
        adjustments.append(Adjustment(
            "capability_floor",
            -tuning["penalty"] * shortfall,
            f"below the quality floor ({quality:.2f} < {floor:.2f}) for a {task.complexity}-complexity task",
        ))
    elif floor > 0:
        bonus = min(MAX_HEADROOM_BONUS, HEADROOM_BONUS_PER_POINT * (quality - floor))
        if bonus > 0:
            adjustments.append(Adjustment(
                "capability_headroom", bonus, f"comfortable headroom for a {task.complexity}-complexity task"
            ))

    if task.privacy_level == "high":
        if model.get("local"):
            adjustments.append(Adjustment(
                "sensitive_local", 0.12, "keeps a sensitive prompt on local hardware"
            ))
        else:
            adjustments.append(Adjustment(
                "sensitive_external", -0.25, "sends a sensitive prompt to an external provider"
            ))

    if health.state == HALF_OPEN:
        adjustments.append(Adjustment(
            "circuit_half_open", -0.20, f"recently failing provider under probe ({health.detail})"
        ))
    elif health.samples >= 5 and health.ok_rate < 1.0:
        penalty = -0.30 * (1.0 - health.ok_rate)
        adjustments.append(Adjustment(
            "observed_failures", penalty, f"observed success rate {health.ok_rate:.0%} over {health.samples} runs"
        ))

    max_context = int(model.get("max_context", 0) or 0)
    if max_context and task.context_tokens > max_context * CONTEXT_PRESSURE_RATIO:
        adjustments.append(Adjustment(
            "context_pressure",
            -0.15,
            f"prompt uses {task.context_tokens / max_context:.0%} of a {max_context} token window",
        ))

    bonus = float(model.get("policy_score_bonus", 0) or 0)
    if bonus:
        adjustments.append(Adjustment("policy_bonus", bonus, "policy prefers local providers"))

    return tuple(adjustments)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
