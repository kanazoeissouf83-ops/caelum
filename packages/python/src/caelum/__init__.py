"""CAELUM — the control plane for AI inference.

Drop-in replacements for provider SDK clients::

    from caelum import OpenAI       # instead of: from openai import OpenAI
    from caelum import AsyncOpenAI  # instead of: from openai import AsyncOpenAI
    from caelum import genai        # instead of: from google import genai
    from caelum import Mistral      # instead of: from mistralai.client import Mistral

Provider SDKs are optional extras (``pip install "caelum-ai[openai]"``). They are imported
lazily, so ``import caelum`` works even when a given provider SDK is not installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from caelum import pricing
from caelum.cache.semantic import OpenAIEmbedder
from caelum.cache.verifier import LLMVerifier
from caelum.cloud import customer
from caelum.context import cache_bypass, language
from caelum.fallback import FallbackChain
from caelum.metrics import Metrics
from caelum.retry import RetryPolicy
from caelum.router import RouterConfig
from caelum.utils import cache_hit, cache_type, routed_to

__version__ = "0.1.0"

# Public name -> (module, attribute). Resolved on first access (PEP 562).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "OpenAI": ("caelum.openai", "CaelumOpenAI"),
    "AsyncOpenAI": ("caelum.openai", "AsyncCaelumOpenAI"),
    "Anthropic": ("caelum.anthropic", "CaelumAnthropic"),
    "AsyncAnthropic": ("caelum.anthropic", "AsyncCaelumAnthropic"),
    "Mistral": ("caelum.mistral", "CaelumMistral"),
    "genai": ("caelum.genai", ""),  # a module: `from caelum import genai`
}

__all__ = [
    "Anthropic",
    "AsyncAnthropic",
    "AsyncOpenAI",
    "FallbackChain",
    "LLMVerifier",
    "Metrics",
    "Mistral",
    "OpenAI",
    "OpenAIEmbedder",
    "RetryPolicy",
    "RouterConfig",
    "__version__",
    "cache_bypass",
    "cache_hit",
    "cache_type",
    "customer",
    "genai",
    "language",
    "pricing",
    "routed_to",
]

if TYPE_CHECKING:
    from caelum import genai as genai
    from caelum.anthropic import AsyncCaelumAnthropic as AsyncAnthropic
    from caelum.anthropic import CaelumAnthropic as Anthropic
    from caelum.mistral import CaelumMistral as Mistral
    from caelum.openai import AsyncCaelumOpenAI as AsyncOpenAI
    from caelum.openai import CaelumOpenAI as OpenAI


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'caelum' has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(module_name)
    value = getattr(module, attribute) if attribute else module
    globals()[name] = value  # cache for next access
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
