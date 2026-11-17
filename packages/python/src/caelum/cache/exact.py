"""Exact cache: identical requests get the stored response instead of a new API call.

The cache key is a SHA-256 hash of every request parameter that can influence the
answer (model, messages, temperature, tools, response_format, seed...). Only
transport-level parameters (timeouts, headers, tracing identifiers) are ignored.

WHY hash everything rather than a short allow-list: two requests that differ only by
``tools`` or ``response_format`` must never share an answer. A missed cache hit costs
a little money; a wrong cache hit breaks the user's product.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from caelum.cache.stores import CacheStore
from caelum.utils import UnhashableRequestError, stable_hash, to_canonical

__all__ = ["EXCLUDED_PARAMS", "ExactCache", "compute_cache_key", "should_cache"]

# Bump when the key format changes, so old entries are never misread.
KEY_VERSION = 2  # 2: customer, language, model and instructions added (0.2)

# Parameters that do not change the generated answer.
EXCLUDED_PARAMS = frozenset(
    {
        "timeout",
        "extra_headers",
        "user",
        "metadata",
        "store",
        "stream_options",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "cache_control",
        "diagnostics",
        # Mistral SDK per-call transport options.
        "retries",
        "timeout_ms",
        "http_headers",
    }
)


def should_cache(request: Mapping[str, Any]) -> bool:
    """Decide whether a request is eligible for caching.

    Streams are not cached (V1). ``tool_choice="required"`` is excluded because the
    caller explicitly expects a fresh tool decision.
    """
    if request.get("stream") is True:
        return False
    return request.get("tool_choice") != "required"


BETA_HEADERS = frozenset({"anthropic-beta", "openai-beta"})


def beta_headers(headers: Any) -> dict[str, str]:
    """Answer-changing beta headers among ``headers`` (names lower-cased)."""
    if not isinstance(headers, Mapping):
        return {}
    items = cast("Mapping[Any, Any]", headers).items()
    return {
        str(name).lower(): str(value)
        for name, value in items
        if str(name).lower() in BETA_HEADERS and isinstance(value, str)
    }


def compute_cache_key(
    request: Mapping[str, Any], *, provider: str, endpoint: str, namespace: str
) -> str | None:
    """Compute the deterministic cache key of a request.

    Args:
        request: Keyword arguments passed to the SDK method.
        provider: Provider name (``"openai"``, ``"anthropic"``...).
        endpoint: Endpoint name (``"chat.completions"``...).
        namespace: Upstream identity (base URL), so different servers never collide.

    Returns:
        A 64-character hex digest, or None if the request cannot be hashed safely
        (in which case it is simply not cached).
    """
    relevant = {
        name: value
        for name, value in request.items()
        if name not in EXCLUDED_PARAMS and not (name == "stream" and value is False)
    }
    betas = beta_headers(request.get("extra_headers"))
    if betas:
        # WHY: beta headers can change the answer (new tool or thinking behaviour).
        relevant["__beta_headers__"] = betas
    try:
        canonical = to_canonical(relevant)
    except UnhashableRequestError:
        return None
    return stable_hash(
        {
            "v": KEY_VERSION,
            "provider": provider,
            "endpoint": endpoint,
            "namespace": namespace,
            "request": canonical,
        }
    )


class ExactCache:
    """Exact-match response cache on top of a :class:`CacheStore`.

    Args:
        store: Storage backend.
        ttl: Entry lifetime in seconds (None = never expires).
    """

    def __init__(self, store: CacheStore, ttl: float | None) -> None:
        if ttl is not None and ttl <= 0:
            raise ValueError("cache_ttl must be a positive number of seconds, or None")
        self.store = store
        self.ttl = ttl

    def get(self, key: str, max_age: float | None = None, owner: str = "") -> dict[str, Any] | None:
        """Return the stored response payload for ``key``, if any.

        ``max_age`` (seconds) applies a lifetime shortened since the entry was written,
        with stores that know when entries were written (custom stores ignore it).
        """
        if max_age is not None and getattr(self.store, "supports_owner", False):
            value = cast(Any, self.store).get(key, max_age=max_age, owner=owner)
        else:
            value = self.store.get(key)
        return cast("dict[str, Any]", value) if isinstance(value, dict) else None

    def set(
        self,
        key: str,
        payload: Mapping[str, Any],
        ttl: float | None = None,
        owner: str = "",
    ) -> None:
        """Store a response payload under ``key`` (for ``ttl`` seconds, default the
        cache's lifetime), labelled with its end customer ``owner``."""
        lifetime = self.ttl if ttl is None else ttl
        if getattr(self.store, "supports_owner", False):
            cast(Any, self.store).set(key, dict(payload), lifetime, owner=owner)
        else:
            self.store.set(key, dict(payload), lifetime)

    def delete(self, key: str) -> None:
        """Remove an entry (e.g. one that can no longer be parsed)."""
        self.store.delete(key)
