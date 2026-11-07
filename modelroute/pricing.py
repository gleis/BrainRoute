"""Token pricing and cost scoring.

Cost scores used to be hand-tuned constants in ``models.yaml``, which meant the
budget policy estimated $0 for every seeded model and never fired. Prices now
come from an explicit table (or the provider's own catalog metadata), and the
0-1 ``cost_score`` used by the router is derived from those prices.

All prices are USD per 1M tokens.
"""

from __future__ import annotations

import math
from typing import Any


# Blended-price endpoints for the cost curve. A model at or below CHEAPEST
# scores 1.0; a model at or above PRICIEST scores near 0.
CHEAPEST_PER_MTOK = 0.05
PRICIEST_PER_MTOK = 75.0

# Output tokens are usually a minority of the blended spend for routing
# workloads, but they are priced far higher, so they carry real weight.
INPUT_WEIGHT = 0.75
OUTPUT_WEIGHT = 0.25

# Typical completion length, in tokens, used when estimating the cost of a
# request before it runs.
ASSUMED_OUTPUT_TOKENS = 600


# Known list prices, keyed by provider-qualified model name. Discovered models
# carry their own pricing when the source reports it and need no entry here.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.4": (1.25, 10.00),
    "gpt-5.4-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    # Generic placeholder tier used by the seed catalog
    "premium-reasoner": (3.00, 15.00),
}


def lookup_price(model: dict[str, Any]) -> tuple[float, float] | None:
    """Return (input, output) USD per 1M tokens, or None if unknown."""
    explicit_in = model.get("input_price_per_mtok")
    explicit_out = model.get("output_price_per_mtok")
    if explicit_in is not None or explicit_out is not None:
        return _float(explicit_in), _float(explicit_out)

    # Discovered catalogs usually express price per single token.
    per_token_in = model.get("input_cost_per_token")
    per_token_out = model.get("output_cost_per_token")
    if per_token_in is not None or per_token_out is not None:
        return _float(per_token_in) * 1_000_000, _float(per_token_out) * 1_000_000

    if model.get("local"):
        return 0.0, 0.0

    for key in (model.get("model"), model.get("id")):
        if isinstance(key, str) and key in PRICE_TABLE:
            return PRICE_TABLE[key]
    return None


def blended_price(input_price: float, output_price: float) -> float:
    return INPUT_WEIGHT * input_price + OUTPUT_WEIGHT * output_price


def cost_score(model: dict[str, Any]) -> float:
    """Map a model's price onto a 0-1 scale where 1.0 is cheapest.

    A logarithmic curve is used because prices span three orders of magnitude;
    a linear map would compress every non-frontier model into the same score.
    """
    price = lookup_price(model)
    if price is None:
        # Unknown pricing is treated as mid-market rather than free, so an
        # unpriced model never wins the cheap profile by default.
        return 0.5
    blended = blended_price(*price)
    if blended <= CHEAPEST_PER_MTOK:
        return 1.0
    span = math.log10(PRICIEST_PER_MTOK / CHEAPEST_PER_MTOK)
    position = math.log10(blended / CHEAPEST_PER_MTOK) / span
    return round(max(0.02, min(1.0, 1.0 - position)), 4)


def apply_pricing(model: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``model`` with pricing fields normalised and filled in."""
    enriched = dict(model)
    price = lookup_price(model)
    if price is not None:
        input_price, output_price = price
        enriched["input_price_per_mtok"] = input_price
        enriched["output_price_per_mtok"] = output_price
        enriched["input_cost_per_token"] = input_price / 1_000_000
        enriched["output_cost_per_token"] = output_price / 1_000_000
        enriched["pricing_known"] = True
    else:
        enriched.setdefault("input_cost_per_token", 0.0)
        enriched.setdefault("output_cost_per_token", 0.0)
        enriched["pricing_known"] = False
    if "cost_score" not in enriched or enriched.get("cost_score") is None:
        enriched["cost_score"] = cost_score(model)
    return enriched


def estimate_request_cost(
    model: dict[str, Any],
    input_tokens: int,
    output_tokens: int = ASSUMED_OUTPUT_TOKENS,
) -> float:
    """Estimated USD for a request, including the completion we expect to pay for."""
    price = lookup_price(model)
    if price is None:
        return 0.0
    input_price, output_price = price
    return round(
        (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price,
        8,
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
