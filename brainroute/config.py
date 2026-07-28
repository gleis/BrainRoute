from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import catalog as catalog_module
from .cache import TTLCache
from .catalog import discover_models, overlay_models
from .config_paths import ROOT
from .settings import settings_version
from .simple_yaml import load_simple_yaml


DEFAULT_MODELS_PATH = ROOT / "config" / "models.yaml"
DEFAULT_WEIGHTS_PATH = ROOT / "config" / "router.weights.yaml"

# The five scoring dimensions. Weights are normalised against this set so that
# profiles can be written with intuitive numbers that need not sum to 1.
FACTORS = ("quality", "speed", "cost", "privacy", "task_fit")

_CONFIG_CACHE = TTLCache(ttl_seconds=30.0, max_entries=16)


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

    def model(self, model_id: str) -> dict[str, Any]:
        for model in self.models:
            if model.get("id") == model_id:
                return model
        raise ValueError(f"Unknown model '{model_id}'")


def load_config(
    models_path: str | Path = DEFAULT_MODELS_PATH,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    discover: bool = False,
) -> RouterConfig:
    """Load and assemble router configuration.

    Cached on (paths, discovery flag, settings version, file mtimes) so that a
    hot request path costs a dict lookup instead of two YAML parses plus live
    provider discovery. Editing a config file or changing settings invalidates
    the entry immediately.
    """
    key = (
        str(models_path),
        str(weights_path),
        bool(discover),
        settings_version(),
        _mtime(models_path),
        _mtime(weights_path),
    )
    return _CONFIG_CACHE.get_or_set(
        key, lambda: _load_uncached(models_path, weights_path, discover)
    )


def invalidate() -> None:
    _CONFIG_CACHE.clear()
    catalog_module.invalidate()


def normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    """Scale a profile's weights so they sum to 1 across the scoring factors.

    Raw profiles in ``router.weights.yaml`` are authored for readability and do
    not sum to 1. Without normalisation, a profile that happened to use larger
    numbers would produce larger scores, making scores incomparable between
    profiles and making bonus terms behave inconsistently between profiles.
    """
    weights = {factor: max(0.0, float(profile.get(factor, 0.0))) for factor in FACTORS}
    total = sum(weights.values())
    if total <= 0:
        return {factor: 1.0 / len(FACTORS) for factor in FACTORS}
    return {factor: value / total for factor, value in weights.items()}


def _load_uncached(
    models_path: str | Path, weights_path: str | Path, discover: bool
) -> RouterConfig:
    model_data = load_simple_yaml(models_path)
    weight_data = load_simple_yaml(weights_path)
    discovered = discover_models()["models"] if discover else []
    profiles = {
        name: dict(values)
        for name, values in dict(weight_data.get("profiles", {})).items()
        if isinstance(values, dict)
    }
    if not profiles:
        raise ValueError(f"No routing profiles defined in {weights_path}")
    return RouterConfig(
        models=overlay_models(list(model_data.get("models", [])), discovered),
        profiles=profiles,
        default_profile=str(weight_data.get("default_profile", next(iter(profiles)))),
    )


def _mtime(path: str | Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0
