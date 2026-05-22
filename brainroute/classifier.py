from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .providers.ollama import call_ollama_json


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    complexity: str
    privacy_level: str
    urgency: str
    signals: tuple[str, ...]
    capabilities: tuple[str, ...] = ()
    context_tokens: int = 0
    confidence: float = 0.5
    classifier: str = "heuristic"
    detail: str = ""


KEYWORDS = {
    "coding": {"code", "bug", "refactor", "python", "swift", "typescript", "api", "test", "debug"},
    "writing": {"write", "rewrite", "tone", "copy", "email", "blog", "headline"},
    "summarization": {"summarize", "summary", "extract", "brief"},
    "legal": {"contract", "legal", "compliance", "terms", "policy"},
    "medical": {"medical", "diagnosis", "patient", "clinical", "symptom"},
    "reasoning": {"plan", "analyze", "architecture", "strategy", "compare", "decide"},
    "tool_use": {"tool", "browser", "shell", "files", "repo", "run"},
}

PRIVATE_HINTS = {"private", "secret", "confidential", "patient", "medical", "internal", "proprietary"}
URGENT_HINTS = {"urgent", "asap", "quick", "quickly", "fast", "now", "time crunch"}
COMPLEX_HINTS = {"architecture", "multi-step", "complex", "deep", "thorough", "production", "long"}


def classify(prompt: str) -> TaskProfile:
    words = _words(prompt)
    scores = {
        task: len(words & hints)
        for task, hints in KEYWORDS.items()
    }
    task_type = max(scores, key=lambda task: (scores[task], task))
    if scores[task_type] == 0:
        task_type = "general"

    complexity = "high" if words & COMPLEX_HINTS or len(prompt) > 1200 else "medium"
    if len(prompt) < 160 and not (words & COMPLEX_HINTS):
        complexity = "low"

    privacy_level = "high" if words & PRIVATE_HINTS else "normal"
    urgency = "high" if words & URGENT_HINTS else "normal"
    signals = tuple(sorted({task_type, complexity, privacy_level, urgency}))
    return TaskProfile(task_type, complexity, privacy_level, urgency, signals, context_tokens=max(1, len(prompt) // 4))


def classify_with_model(prompt: str, model: dict[str, Any]) -> TaskProfile:
    payload = call_ollama_json(model, _router_prompt(prompt), ROUTER_SCHEMA, timeout=20)
    heuristic = classify(prompt)
    task_type = _allowed(payload.get("task_type"), set(KEYWORDS) | {"general"}, heuristic.task_type)
    complexity = _allowed(payload.get("complexity"), {"low", "medium", "high"}, heuristic.complexity)
    privacy = _allowed(payload.get("privacy_level"), {"normal", "high"}, heuristic.privacy_level)
    urgency = _allowed(payload.get("urgency"), {"normal", "high"}, heuristic.urgency)
    capabilities = tuple(
        item for item in payload.get("capabilities", [])
        if isinstance(item, str) and item[:60]
    )[:8]
    return TaskProfile(
        task_type=task_type,
        complexity=complexity,
        privacy_level=privacy,
        urgency=urgency,
        signals=tuple(sorted({task_type, complexity, privacy, urgency, "model"})),
        capabilities=capabilities,
        context_tokens=_integer(payload.get("context_tokens"), heuristic.context_tokens),
        confidence=_confidence(payload.get("confidence")),
        classifier=f"model:{model['id']}",
        detail=str(payload.get("reason", ""))[:240],
    )


def _router_prompt(prompt: str) -> str:
    return (
        "Classify this user request for model routing. Return only JSON matching the schema. "
        "Mark privacy_level high for secrets, private business data, medical, legal-client, "
        "credentials, or personal records. Estimate context_tokens for the request itself.\n\n"
        f"Request:\n{prompt}"
    )


ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": [*KEYWORDS, "general"]},
        "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "privacy_level": {"type": "string", "enum": ["normal", "high"]},
        "urgency": {"type": "string", "enum": ["normal", "high"]},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "context_tokens": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["task_type", "complexity", "privacy_level", "urgency", "capabilities", "context_tokens", "confidence", "reason"],
}


def _allowed(value: Any, options: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in options else default


def _integer(value: Any, default: int) -> int:
    try:
        return max(1, min(int(value), 10000000))
    except (TypeError, ValueError):
        return default


def _confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(float(value), 1.0)), 3)
    except (TypeError, ValueError):
        return 0.5


def _words(prompt: str) -> set[str]:
    lowered = prompt.lower().replace("-", " ")
    return {"".join(ch for ch in word if ch.isalnum() or ch == "_") for word in lowered.split()}
