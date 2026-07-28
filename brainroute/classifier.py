"""Request classification.

Two classifiers share one output shape. The heuristic is always available and
costs microseconds; the structured local-model classifier is more accurate and
costs a round trip to Ollama, so its results are cached and it always degrades
back to the heuristic rather than failing a request.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from .cache import TTLCache
from .providers.ollama import call_ollama_json
from .settings import load_settings


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "privacy_level": self.privacy_level,
            "urgency": self.urgency,
            "signals": list(self.signals),
            "capabilities": list(self.capabilities),
            "context_tokens": self.context_tokens,
            "confidence": self.confidence,
            "classifier": self.classifier,
            "detail": self.detail,
        }


KEYWORDS = {
    "coding": {"code", "bug", "refactor", "python", "swift", "typescript", "javascript",
               "api", "test", "debug", "function", "compile", "stacktrace", "regression"},
    "writing": {"write", "rewrite", "tone", "copy", "email", "blog", "headline",
                "draft", "post", "caption", "newsletter"},
    "summarization": {"summarize", "summary", "extract", "brief", "tldr", "condense", "digest"},
    "legal": {"contract", "legal", "compliance", "terms", "policy", "clause",
              "liability", "indemnity", "gdpr"},
    "medical": {"medical", "diagnosis", "patient", "clinical", "symptom",
                "dosage", "prognosis"},
    "reasoning": {"plan", "analyze", "analyse", "architecture", "strategy", "compare",
                  "decide", "tradeoff", "evaluate", "design", "why"},
    "tool_use": {"tool", "browser", "shell", "files", "repo", "run", "scrape", "invoke"},
}

PRIVATE_HINTS = {"private", "secret", "confidential", "patient", "medical", "internal",
                 "proprietary", "nda", "ssn", "password", "credentials", "payroll", "salary"}
URGENT_HINTS = {"urgent", "asap", "quick", "quickly", "fast", "now", "immediately", "rush"}
COMPLEX_HINTS = {"architecture", "complex", "deep", "thorough", "production", "comprehensive",
                 "end-to-end", "distributed", "migrate", "refactor"}

# Multi-word signals cannot be matched against a bag of words, so they are
# checked as substrings of the normalised prompt.
URGENT_PHRASES = ("time crunch", "right away", "as soon as possible", "by end of day")
COMPLEX_PHRASES = ("multi-step", "multi step", "step by step", "from scratch", "trade-off")
PRIVATE_PHRASES = ("do not share", "internal only", "personal data", "under nda")

LONG_PROMPT_CHARS = 1200
SHORT_PROMPT_CHARS = 160
CHARS_PER_TOKEN = 4

_CLASSIFIER_CACHE = TTLCache(ttl_seconds=300.0, max_entries=512)


def classify(prompt: str) -> TaskProfile:
    """Fast, dependency-free classification. Always succeeds."""
    lowered = prompt.lower()
    words = _words(lowered)

    scores = {task: len(words & hints) for task, hints in KEYWORDS.items()}
    task_type = max(scores, key=lambda task: (scores[task], task))
    if scores[task_type] == 0:
        task_type = "general"

    complex_signal = bool(words & COMPLEX_HINTS) or _has_phrase(lowered, COMPLEX_PHRASES)
    if complex_signal or len(prompt) > LONG_PROMPT_CHARS:
        complexity = "high"
    elif len(prompt) < SHORT_PROMPT_CHARS:
        complexity = "low"
    else:
        complexity = "medium"

    privacy_level = "high" if (words & PRIVATE_HINTS or _has_phrase(lowered, PRIVATE_PHRASES)) else "normal"
    urgency = "high" if (words & URGENT_HINTS or _has_phrase(lowered, URGENT_PHRASES)) else "normal"

    # Confidence reflects how much evidence the keyword match rested on.
    matches = scores.get(task_type, 0)
    confidence = 0.4 if task_type == "general" else round(min(0.85, 0.5 + 0.12 * matches), 3)

    signals = tuple(sorted({task_type, complexity, privacy_level, urgency}))
    return TaskProfile(
        task_type=task_type,
        complexity=complexity,
        privacy_level=privacy_level,
        urgency=urgency,
        signals=signals,
        context_tokens=estimate_tokens(prompt),
        confidence=confidence,
    )


def classify_with_model(prompt: str, model: dict[str, Any], use_cache: bool = True) -> TaskProfile:
    """Structured classification via a local router model, with a heuristic floor.

    Every field is validated against the heuristic result, so a malformed or
    hallucinated response degrades to the heuristic value rather than producing
    an invalid task profile.
    """
    if not use_cache:
        return _classify_with_model_uncached(prompt, model)
    ttl = float(load_settings()["classifier"].get("cache_ttl_seconds", 300) or 0)
    if ttl <= 0:
        return _classify_with_model_uncached(prompt, model)
    _CLASSIFIER_CACHE.ttl_seconds = ttl
    key = (model.get("id"), _fingerprint(prompt))
    return _CLASSIFIER_CACHE.get_or_set(key, lambda: _classify_with_model_uncached(prompt, model))


def classify_prompt(prompt: str, model: dict[str, Any] | None) -> TaskProfile:
    """Classify with the router model when configured, else heuristically.

    Model failures are absorbed here: the heuristic result is returned with the
    failure recorded in ``detail`` so the operator can see the degradation.
    """
    if model is None:
        return classify(prompt)
    try:
        return classify_with_model(prompt, model)
    except Exception as exc:  # noqa: BLE001 - any classifier failure must degrade, not raise
        return replace(classify(prompt), detail=f"router model unavailable: {exc}"[:240])


def invalidate() -> None:
    _CLASSIFIER_CACHE.clear()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def _classify_with_model_uncached(prompt: str, model: dict[str, Any]) -> TaskProfile:
    payload = call_ollama_json(model, _router_prompt(prompt), ROUTER_SCHEMA, timeout=20)
    heuristic = classify(prompt)
    task_type = _allowed(payload.get("task_type"), set(KEYWORDS) | {"general"}, heuristic.task_type)
    complexity = _allowed(payload.get("complexity"), {"low", "medium", "high"}, heuristic.complexity)
    privacy = _allowed(payload.get("privacy_level"), {"normal", "high"}, heuristic.privacy_level)
    urgency = _allowed(payload.get("urgency"), {"normal", "high"}, heuristic.urgency)
    capabilities = tuple(
        item.strip()[:60]
        for item in payload.get("capabilities", [])
        if isinstance(item, str) and item.strip()
    )[:8]
    # A model that misses a privacy or urgency signal the heuristic caught is
    # overridden: under-detecting either is the costlier error.
    if heuristic.privacy_level == "high":
        privacy = "high"
    if heuristic.urgency == "high":
        urgency = "high"
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


def _fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _has_phrase(lowered: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in lowered for phrase in phrases)


def _allowed(value: Any, options: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in options else default


def _integer(value: Any, default: int) -> int:
    try:
        return max(1, min(int(value), 10_000_000))
    except (TypeError, ValueError):
        return default


def _confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(float(value), 1.0)), 3)
    except (TypeError, ValueError):
        return 0.5


def _words(lowered: str) -> set[str]:
    normalised = lowered.replace("-", " ")
    return {
        "".join(char for char in word if char.isalnum() or char == "_")
        for word in normalised.split()
    }
