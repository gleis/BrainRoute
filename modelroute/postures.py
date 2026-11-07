"""Privacy postures.

A posture is a named, coarse-grained stance on where data is allowed to go. It
exists so the common decision - "how much do I trust cloud providers today" -
is one setting rather than four correlated flags a user has to keep consistent.

Postures are a *base layer*, not a lock. Any individual policy flag a user sets
explicitly overrides what the posture would have chosen, so a posture is a
starting point you can tune rather than a mode you are trapped in. The one
exception is ``air_gapped``, which deliberately cannot be overridden per
request - a posture that a single API call can switch off is not a guarantee.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_POSTURE = "local_first"


POSTURES: dict[str, dict[str, Any]] = {
    "open": {
        "label": "Open",
        "description": (
            "Cloud and local models compete on equal terms. Sensitive prompts "
            "still stay local unless you explicitly allow otherwise."
        ),
        "settings": {
            "policy": {
                "prefer_local": False,
                "allow_cloud": True,
                "allow_cloud_for_private": False,
            },
        },
        "default_profile": None,
        "cloud_override_allowed": True,
        "remote_discovery_allowed": True,
    },
    "local_first": {
        "label": "Local first",
        "description": (
            "Local models get a scoring bonus and win by default. Cloud stays "
            "available and takes over only when no local model can do the job - "
            "and the router tells you when that happens."
        ),
        # Deliberately does NOT force the private profile. Doing so would make
        # the privacy weight so dominant that the capability floor could never
        # escalate a genuinely hard task, which is the opposite of "cloud wins
        # only when local cannot do the job".
        "settings": {
            "policy": {
                "prefer_local": True,
                "allow_cloud": True,
                "allow_cloud_for_private": False,
            },
        },
        "default_profile": None,
        "cloud_override_allowed": True,
        "remote_discovery_allowed": True,
    },
    "local_only": {
        "label": "Local only",
        "description": (
            "Cloud providers are rejected by policy. A caller can still opt in "
            "per request, so escalation is possible but never accidental."
        ),
        "settings": {
            "policy": {
                "prefer_local": True,
                "allow_cloud": False,
                "allow_cloud_for_private": False,
            },
        },
        "default_profile": "private",
        "cloud_override_allowed": True,
        "remote_discovery_allowed": True,
    },
    "air_gapped": {
        "label": "Air gapped",
        "description": (
            "No outbound traffic except to your own local runtime. Cloud "
            "providers are rejected, remote catalog discovery is disabled, and "
            "no per-request override can re-enable either."
        ),
        "settings": {
            "policy": {
                "prefer_local": True,
                "allow_cloud": False,
                "allow_cloud_for_private": False,
            },
        },
        "default_profile": "private",
        "cloud_override_allowed": False,
        # Listing remote catalogs is outbound traffic even though it carries no
        # prompt data, so air-gapped blocks it outright rather than relying on
        # the preset - which a pinned setting would otherwise override.
        "remote_discovery_allowed": False,
    },
}


def resolve(name: Any) -> str:
    """Return a known posture name, falling back to the default."""
    return name if isinstance(name, str) and name in POSTURES else DEFAULT_POSTURE


def preset(name: Any) -> dict[str, Any]:
    """The settings layer a posture contributes, as a deep copy."""
    return deepcopy(POSTURES[resolve(name)]["settings"])


def default_profile(name: Any) -> str | None:
    return POSTURES[resolve(name)]["default_profile"]


def cloud_override_allowed(name: Any) -> bool:
    return bool(POSTURES[resolve(name)]["cloud_override_allowed"])


def remote_discovery_allowed(name: Any) -> bool:
    """Whether this posture permits contacting a remote model catalog at all."""
    return bool(POSTURES[resolve(name)]["remote_discovery_allowed"])


def describe() -> list[dict[str, Any]]:
    """Posture catalog for the UI and the CLI."""
    return [
        {
            "id": key,
            "label": value["label"],
            "description": value["description"],
            "default_profile": value["default_profile"],
            "cloud_override_allowed": value["cloud_override_allowed"],
            "remote_discovery_allowed": value["remote_discovery_allowed"],
            "policy": value["settings"].get("policy", {}),
        }
        for key, value in POSTURES.items()
    ]
