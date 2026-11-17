"""Multi-provider automatic fallback.

When the primary request fails with a transient error (rate limit, 5xx, overload,
timeout, network error) — after CAELUM retries, if enabled — CAELUM tries each fallback
tier in order, translating the request to the tier's provider when needed. Your code
receives a response in the format it asked for, as if nothing happened::

    client = OpenAI(
        fallback=["anthropic:claude-sonnet-5-5", "openai:gpt-4o-mini"],
        provider_keys={"anthropic": "sk-ant-..."},   # or ANTHROPIC_API_KEY in the env
    )

Rules:

* each tier names a provider AND a model: CAELUM never picks a model for you
  (neutrality, and the quality contract stays yours);
* every provider needs its own API key (CAELUM does not resell tokens);
* client errors (400, 401, 403, 404, 422...) never trigger a fallback: they would fail
  everywhere and hide a bug;
* fallback answers are not stored in the caches (the next request retries the primary);
* local backends (``"ollama:qwen3:1.7b"``, ``llamacpp``, ``vllm``, ``lmstudio``) need no
  key and cost nothing: a good last tier when the cloud is down;
* streaming requests only fall back to tiers speaking the same API format (OpenAI-format
  clients can stream from a local backend; OpenAI <-> Anthropic streaming is not
  translated yet);
* when every tier fails, the ORIGINAL error is re-raised.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from caelum.providers import SUPPORTED_PROVIDERS, dialect, is_local, local_base_url
from caelum.retry import RetryPolicy, is_retryable
from caelum.utils import missing_extra_message

__all__ = ["FallbackChain", "FallbackTier", "ProviderPool", "should_fallback"]

_DEFAULT_POLICY = RetryPolicy()


@dataclass(frozen=True)
class FallbackTier:
    """One fallback destination."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _parse_tier(spec: Any) -> FallbackTier:
    if isinstance(spec, FallbackTier):
        return spec
    if isinstance(spec, str):
        provider, sep, model = spec.partition(":")
        if not sep or not model:
            raise ValueError(
                f"fallback tier {spec!r} must name a model, e.g. 'anthropic:claude-sonnet-5-5'"
                " (CAELUM never chooses a model on your behalf)"
            )
    elif isinstance(spec, Mapping):
        mapping = cast("Mapping[str, Any]", spec)
        unknown = set(mapping) - {"provider", "model"}
        if unknown:
            raise ValueError(f"fallback tier: unknown key(s) {sorted(unknown)}")
        provider, model = mapping.get("provider"), mapping.get("model")
    else:
        raise TypeError(f"invalid fallback tier: {spec!r}")
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"unsupported fallback provider {provider!r} (supported: {supported})")
    if not isinstance(model, str) or not model:
        raise ValueError("fallback tier model must be a non-empty string")
    return FallbackTier(provider=cast(str, provider), model=model)


@dataclass(frozen=True)
class FallbackChain:
    """Ordered fallback tiers tried after the primary request fails."""

    tiers: tuple[FallbackTier, ...]

    @classmethod
    def parse(cls, spec: FallbackChain | Iterable[Any]) -> FallbackChain:
        """Build a chain from tiers like ``"anthropic:claude-sonnet-5-5"`` or dicts."""
        if isinstance(spec, FallbackChain):
            return spec
        if isinstance(spec, (str, bytes)):
            raise TypeError("fallback must be a list of tiers, not a single string")
        tiers = tuple(_parse_tier(item) for item in spec)
        if not tiers:
            raise ValueError("fallback needs at least one tier")
        return cls(tiers=tiers)

    @property
    def providers(self) -> frozenset[str]:
        """Every provider referenced by the chain."""
        return frozenset(tier.provider for tier in self.tiers)


def should_fallback(exc: BaseException) -> bool:
    """True for transient failures (429, 5xx, 529, timeouts, network errors)."""
    return is_retryable(exc, _DEFAULT_POLICY)


_SDK_CLASSES = {
    ("openai", False): ("openai", "OpenAI"),
    ("openai", True): ("openai", "AsyncOpenAI"),
    ("anthropic", False): ("anthropic", "Anthropic"),
    ("anthropic", True): ("anthropic", "AsyncAnthropic"),
}


class ProviderPool:
    """Raw SDK clients for providers other than the wrapped one (created lazily).

    Args:
        native: Provider of the wrapped client.
        is_async: Whether async SDK clients are needed.
        keys: API keys per provider (default: the SDK's own environment variable).
        clients: Ready-made raw SDK clients per provider (advanced, e.g. custom base URL).
        sdk_retries: False when CAELUM retries are on, so that the SDK's own retries
            are disabled on the clients created here and attempts never multiply.
    """

    def __init__(
        self,
        *,
        native: str,
        is_async: bool,
        keys: Mapping[str, str] | None = None,
        clients: Mapping[str, Any] | None = None,
        sdk_retries: bool = True,
    ) -> None:
        self.native = native
        self.is_async = is_async
        self.sdk_retries = sdk_retries
        self._owned: list[Any] = []
        for option, value in (("provider_keys", keys), ("provider_clients", clients)):
            if value is not None and not isinstance(value, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f'{option} must be a dict, e.g. {{"anthropic": ...}}')
        self._keys = dict(keys or {})
        self._clients: dict[str, Any] = dict(clients or {})
        for name in [*self._keys, *self._clients]:
            if name not in SUPPORTED_PROVIDERS:
                raise ValueError(f"unknown provider {name!r} in provider_keys/provider_clients")
        for name, client in self._clients.items():
            client_is_async = type(client).__name__.startswith("Async")
            if client_is_async != is_async:
                wanted = "an async (AsyncOpenAI/AsyncAnthropic)" if is_async else "a sync"
                raise TypeError(f"provider_clients[{name!r}] must be {wanted} SDK client")

    def require(self, providers: Iterable[str]) -> None:
        """Fail early, with an install hint, if a needed provider SDK is missing."""
        for provider in providers:
            if provider == self.native or provider in self._clients:
                continue
            module, _ = _SDK_CLASSES[(dialect(provider), self.is_async)]
            try:
                importlib.import_module(module)
            except ImportError as exc:
                raise ImportError(missing_extra_message(module, "caelum fallback/routing")) from exc

    def get(self, provider: str) -> Any:
        """Return the raw SDK client for ``provider`` (created on first use)."""
        client = self._clients.get(provider)
        if client is None:
            module, cls = _SDK_CLASSES[(dialect(provider), self.is_async)]
            sdk_class = getattr(importlib.import_module(module), cls)
            key = self._keys.get(provider)
            options: dict[str, Any] = {} if self.sdk_retries else {"max_retries": 0}
            if is_local(provider):
                # WHY: local servers ignore the key, but the openai SDK requires one.
                options.update(base_url=local_base_url(provider), api_key=key or "local")
            elif key:
                options["api_key"] = key
            client = sdk_class(**options)
            self._clients[provider] = client
            self._owned.append(client)
        return client

    def close(self) -> None:
        """Close the SDK clients created by the pool (never the ones the user passed)."""
        owned, self._owned = self._owned, []
        for client in owned:
            self._clients = {k: v for k, v in self._clients.items() if v is not client}
            close = getattr(client, "close", None)
            if close is None or self.is_async:
                continue
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                pass

    async def aclose(self) -> None:
        """Asynchronous counterpart of :meth:`close` (for async SDK clients)."""
        owned, self._owned = self._owned, []
        for client in owned:
            self._clients = {k: v for k, v in self._clients.items() if v is not client}
            close = getattr(client, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:  # pragma: no cover - best effort
                pass
