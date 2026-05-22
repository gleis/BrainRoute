from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    complexity: str
    privacy_level: str
    urgency: str
    signals: tuple[str, ...]


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
    return TaskProfile(task_type, complexity, privacy_level, urgency, signals)


def _words(prompt: str) -> set[str]:
    lowered = prompt.lower().replace("-", " ")
    return {"".join(ch for ch in word if ch.isalnum() or ch == "_") for word in lowered.split()}
