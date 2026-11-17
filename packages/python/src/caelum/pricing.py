"""Static price table used to *estimate* request costs.

IMPORTANT: prices are maintained by hand and can become outdated. Always treat the
resulting costs as estimates and check the providers' pricing pages for billing:

* OpenAI: https://openai.com/api/pricing
* Anthropic: https://www.anthropic.com/pricing
* Google Gemini: https://ai.google.dev/gemini-api/docs/pricing
* Mistral: https://docs.mistral.ai/inference/pricing

Prices are in USD per 1 million tokens. Override or add models at runtime with
:func:`register_price`.
"""

from __future__ import annotations

import threading
from typing import TypedDict

__all__ = [
    "PRICING",
    "PRICING_LAST_UPDATED",
    "ModelPrice",
    "estimate_cost",
    "get_price",
    "register_price",
]

PRICING_LAST_UPDATED = "2026-11-02"


class ModelPrice(TypedDict):
    """Price of a model in USD per 1 million tokens."""

    input_per_1m: float
    output_per_1m: float


def _price(input_per_1m: float, output_per_1m: float) -> ModelPrice:
    return {"input_per_1m": input_per_1m, "output_per_1m": output_per_1m}


PRICING: dict[str, ModelPrice] = {
    # --- OpenAI (verify against https://openai.com/api/pricing) ---------------------
    "gpt-5": _price(1.25, 10.00),
    "gpt-5-mini": _price(0.25, 2.00),
    "gpt-5-nano": _price(0.05, 0.40),
    "gpt-4.1": _price(2.00, 8.00),
    "gpt-4.1-mini": _price(0.40, 1.60),
    "gpt-4.1-nano": _price(0.10, 0.40),
    "gpt-4o": _price(2.50, 10.00),
    "gpt-4o-mini": _price(0.15, 0.60),
    "gpt-4-turbo": _price(10.00, 30.00),
    "gpt-3.5-turbo": _price(0.50, 1.50),
    "o3": _price(2.00, 8.00),
    "o4-mini": _price(1.10, 4.40),
    "text-embedding-3-small": _price(0.02, 0.0),
    "text-embedding-3-large": _price(0.13, 0.0),
    "text-embedding-ada-002": _price(0.10, 0.0),
    # --- Anthropic (first-party API rates, as of 2026-10-06) -----------------------
    "claude-fable-5-1": _price(10.00, 50.00),
    "claude-fable-5": _price(10.00, 50.00),
    "claude-opus-5-5": _price(4.00, 20.00),
    "claude-opus-5": _price(5.00, 25.00),
    "claude-opus-4-8": _price(5.00, 25.00),
    "claude-opus-4-7": _price(5.00, 25.00),
    "claude-opus-4-6": _price(5.00, 25.00),
    "claude-sonnet-5-5": _price(2.00, 10.00),
    "claude-sonnet-5": _price(2.00, 10.00),
    "claude-sonnet-4-6": _price(3.00, 15.00),
    # Prompts over 100K tokens are billed $0.50 / $2.50 (not modelled here).
    "claude-haiku-5-5": _price(0.10, 0.50),
    "claude-haiku-4-5": _price(1.00, 5.00),
    # --- Google Gemini (paid tier, text input; checked 2026-11-02 on
    # https://ai.google.dev/gemini-api/docs/pricing). Output includes thinking tokens.
    # Long prompts (> 200K tokens) on Pro models cost more (not modelled here). ---------
    "gemini-3-flash-preview": _price(0.50, 3.00),
    "gemini-3.1-flash-lite": _price(0.25, 1.50),
    "gemini-2.5-pro": _price(1.25, 10.00),
    "gemini-2.5-flash": _price(0.30, 2.50),
    "gemini-2.5-flash-lite": _price(0.10, 0.40),
    # --- Mistral (checked 2026-11-02 on https://docs.mistral.ai/inference/pricing).
    # Base names also match "-latest" aliases and dated versions (current generation:
    # Small 4, Medium 3.5, Large 4). ----------------------------------------------------
    "mistral-small": _price(0.15, 0.60),
    "mistral-medium": _price(1.50, 7.50),
    "mistral-large": _price(0.70, 2.10),
}

_lock = threading.Lock()


def register_price(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Add or override the price of a model (USD per 1M tokens).

    Example::

        caelum.pricing.register_price("my-finetuned-model", 3.0, 12.0)
    """
    if input_per_1m < 0 or output_per_1m < 0:
        raise ValueError("prices must be non-negative")
    with _lock:
        PRICING[model] = _price(input_per_1m, output_per_1m)


def get_price(model: str) -> ModelPrice | None:
    """Return the price of ``model``, matching dated snapshots to their base model.

    ``"gpt-4o-2024-08-06"`` resolves to ``"gpt-4o"`` and ``"gpt-4o-mini-2024-07-18"`` to
    ``"gpt-4o-mini"`` (the longest matching prefix wins).
    """
    with _lock:
        exact = PRICING.get(model)
        if exact is not None:
            return exact
        best: str | None = None
        for name in PRICING:
            if model.startswith(name + "-") and (best is None or len(name) > len(best)):
                best = name
        return PRICING[best] if best is not None else None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate the cost of a request in USD, or None if the model price is unknown."""
    price = get_price(model)
    if price is None:
        return None
    return (
        input_tokens * price["input_per_1m"] + output_tokens * price["output_per_1m"]
    ) / 1_000_000
