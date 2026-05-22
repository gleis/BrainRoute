from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import discover_models, overlay_models
from .config_paths import ROOT
from .simple_yaml import load_simple_yaml


DEFAULT_MODELS_PATH = ROOT / "config" / "models.yaml"
DEFAULT_WEIGHTS_PATH = ROOT / "config" / "router.weights.yaml"


@dataclass(frozen=True)
class RouterConfig:
    models: list[dict[str, Any]]
    profiles: dict[str, dict[str, float]]
    default_profile: str

    def profile(self, name: str | None = None) -> dict[str, float]:
        profile_name = name or self.default_profile
        try:
            return self.profiles[profile_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles))
            raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}") from exc


def load_config(
    models_path: str | Path = DEFAULT_MODELS_PATH,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    discover: bool = False,
) -> RouterConfig:
    model_data = load_simple_yaml(models_path)
    weight_data = load_simple_yaml(weights_path)
    discovered = discover_models()["models"] if discover else []
    return RouterConfig(
        models=overlay_models(list(model_data.get("models", [])), discovered),
        profiles=dict(weight_data.get("profiles", {})),
        default_profile=str(weight_data.get("default_profile", "balanced")),
    )
