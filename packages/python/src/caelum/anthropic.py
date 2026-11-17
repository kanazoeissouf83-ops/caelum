"""Drop-in wrapper for the Anthropic SDK.

Usage::

    from caelum import Anthropic  # instead of: from anthropic import Anthropic

    client = Anthropic(api_key="sk-ant-...")
    client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, messages=[...])

Every constructor argument, method and attribute of ``anthropic.Anthropic`` keeps
working. CAELUM only intercepts ``messages.create``; everything else (``messages.stream``,
``messages.count_tokens``, ``messages.batches``, ``beta``...) is forwarded untouched.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from caelum._engine import (
    DEFAULT_CACHE_TTL,
    CaelumState,
    Endpoint,
    Engine,
    build_state,
    clear_customer_cache,
    sdk_kwargs,
)
from caelum._proxy import (
    AsyncClientProxy,
    AsyncInterceptedResource,
    InterceptedResource,
    SyncClientProxy,
)
from caelum.cache.exact import beta_headers
from caelum.cache.semantic import (
    AsyncEmbedder,
    Embedder,
    anthropic_semantic_query,
)
from caelum.cache.stores import CacheStore
from caelum.cache.verifier import AsyncVerifier, Verifier
from caelum.cloud import customer_tag
from caelum.fallback import FallbackChain
from caelum.journal import Journal
from caelum.metrics import Metrics
from caelum.retry import RetryPolicy, coerce_policy
from caelum.router.config import RouterConfig
from caelum.utils import (
    credential_scope,
    memoized_namespace,
    missing_extra_message,
    stable_hash,
)

try:
    import anthropic
    from anthropic.types import Message
except ImportError as exc:  # pragma: no cover - exercised via a dedicated test
    raise ImportError(missing_extra_message("anthropic")) from exc

if TYPE_CHECKING:
    # WHY: type checkers see the real SDK class, so users keep full typing.
    _SyncBase = anthropic.Anthropic
    _AsyncBase = anthropic.AsyncAnthropic
else:
    _SyncBase = SyncClientProxy
    _AsyncBase = AsyncClientProxy

__all__ = ["AsyncCaelumAnthropic", "CaelumAnthropic"]

_PROVIDER = "anthropic"
_STATE_ATTR = "_caelum_state"
MESSAGES = Endpoint(
    provider="anthropic",
    name="messages",
    response_type=Message,
    semantic_query=anthropic_semantic_query,
    routable=True,
)


def _bind(wrapper: object, client: Any, state: CaelumState, *, is_async: bool) -> None:
    """Attach an SDK client and the intercepted resources to a wrapper instance."""

    def inputs() -> tuple[Any, ...]:
        return (
            str(client.base_url),
            getattr(client, "api_key", None),
            getattr(client, "auth_token", None),
            tuple(sorted(beta_headers(getattr(client, "_custom_headers", None)).items())),
        )

    def build(key: tuple[Any, ...]) -> str:
        base_url, api_key, auth_token, betas = key
        scope = credential_scope(api_key, auth_token)
        return f"{base_url}{scope}{stable_hash(list(betas)) if betas else ''}"

    namespace = memoized_namespace(inputs, build)

    engine = Engine(namespace=namespace, state=state)
    resource = AsyncInterceptedResource if is_async else InterceptedResource
    resources: dict[str, Any] = {"messages": resource(client.messages, engine, MESSAGES)}
    object.__setattr__(wrapper, "_caelum_client", client)
    object.__setattr__(wrapper, _STATE_ATTR, state)
    object.__setattr__(wrapper, "_caelum_engine", engine)
    object.__setattr__(wrapper, "_caelum_resources", resources)


class CaelumAnthropic(_SyncBase):
    """Drop-in replacement for ``anthropic.Anthropic``.

    Accepts exactly the same arguments as ``anthropic.Anthropic``, plus the optional
    CAELUM options below.

    Args:
        *args: Forwarded to ``anthropic.Anthropic``.
        cache: Enable the exact response cache. Disabled by default: without it, CAELUM
            behaves exactly like the Anthropic SDK.
        cache_ttl: Cache entry lifetime in seconds (default 24h, None = no expiry).
        cache_path: SQLite cache file (default ``~/.caelum/cache.sqlite``, or
            ``$CAELUM_HOME/cache.sqlite``).
        cache_store: Custom :class:`~caelum.cache.CacheStore` (advanced).
        retry: Enable CAELUM retries (``True`` or a :class:`~caelum.RetryPolicy`).
            Disabled by default; when enabled, the SDK's own retries are turned off
            unless ``max_retries`` is passed explicitly.
        semantic_cache: Enable the semantic cache (reuse answers to similar questions
            asked in the same context). Disabled by default.
        semantic_threshold: Minimum cosine similarity to reuse an answer (default
            0.60 with a verifier, 0.97 without).
        semantic_gray_threshold: Lower bound of the gray zone (default
            ``semantic_threshold - 0.05``). Between the two bounds the API is called and
            its answer replaces the close-but-uncertain stored entry.
        semantic_threshold_floor: Safety floor (0.80 to 0.999): a per-request
            ``semantic_threshold`` below it is raised to it.
        semantic_customer_thresholds: Per end customer bounds, e.g.
            ``{"acme": (0.98, 0.93)}`` (see :func:`caelum.customer`).
        semantic_verifier: Judge confirming each semantic match before reuse, e.g.
            ``caelum.LLMVerifier(api_key=...)`` (OpenAI ``gpt-4o-mini``). None by
            default for Anthropic: the strict 0.97 threshold is then used.
        semantic_embedder: Required with ``semantic_cache=True``, for example
            ``caelum.OpenAIEmbedder(api_key=...)``. The question text is then sent to
            that embedding provider.
        router_config: Routing rules (dict, ``RouterConfig`` or ``.yaml``/``.json`` path)
            sending each request to the cheapest capable model. Disabled by default.
        fallback: Fallback tiers tried when the provider fails, e.g.
            ``["anthropic:claude-sonnet-5-5"]``. Each provider needs its own API key.
        provider_keys: API keys of the other providers used by routing/fallback (default:
            their SDK environment variables, e.g. ``ANTHROPIC_API_KEY``).
        provider_clients: Ready-made raw SDK clients of other providers (advanced).
        cache_admission: API answers needed for a question before it is cached
            (default 2: savings start with its third request; 1 = at once).
        cache_customer_admission: Per end customer ``cache_admission``, e.g.
            ``{"acme": 3}``.
        cache_exclusions: Automatic exclusion categories to switch off, e.g.
            ``{"amount": False}`` (all on: order, phone, email, amount,
            datetime, payment_identity).
        cache_exclude_patterns: Extra regular expressions that keep a question
            out of the cache.
        cache_customer_exclusions: Per end customer exclusion settings, e.g.
            ``{"acme": {"amount": False, "patterns": [...]}}``.
        journal: Per-request journal (on by default with a cache, stored locally in
            ``~/.caelum/journal.sqlite``); False to turn it off.
        journal_path: Journal file location.
        journal_retention_days: Days a journal line is kept (default 30).
        avoided_call_cost: Estimated cost of an avoided AI call when the real
            token cost is unknown (default $0.002).
        customer_avoided_call_cost: Per end customer ``avoided_call_cost``.
        judge_call_cost: Estimated cost of one judge call when its real tokens are
            unknown (default $0.00006, measured with gpt-4o-mini).
        embedding_call_cost: Estimated cost of one embedding when its real tokens
            are unknown (default $0.000001).
        cache_customer_ttl: Lifetime of cached answers per end customer, in seconds
            (1 minute to 30 days), e.g. ``{"acme": 3600}``. A shorter lifetime
            also applies to answers already cached.
        caelum_key: CAELUM Cloud key (``sk-caelum-...``, default ``$CAELUM_API_KEY``).
            Sends usage metrics (never prompts) to your CAELUM Cloud dashboard and, with
            ``cache=True``, adds the shared Cloud cache behind the local one. Your AI calls
            still go directly to the provider. None (default) = no Cloud at all.
        **kwargs: Forwarded to ``anthropic.Anthropic``.
    """

    def __init__(
        self,
        *args: Any,
        cache: bool = False,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        cache_path: str | os.PathLike[str] | None = None,
        cache_store: CacheStore | None = None,
        retry: bool | RetryPolicy | None = None,
        semantic_cache: bool = False,
        semantic_threshold: float | None = None,
        semantic_gray_threshold: float | None = None,
        semantic_customer_thresholds: Mapping[str, tuple[float, float]] | None = None,
        semantic_threshold_floor: float | None = None,
        semantic_embedder: Embedder | AsyncEmbedder | None = None,
        semantic_verifier: Verifier | AsyncVerifier | bool | None = None,
        router_config: RouterConfig | Mapping[str, Any] | str | os.PathLike[str] | None = None,
        fallback: Sequence[str | Mapping[str, str]] | FallbackChain | None = None,
        provider_keys: Mapping[str, str] | None = None,
        provider_clients: Mapping[str, Any] | None = None,
        caelum_key: str | None = None,
        cache_admission: int | None = None,
        cache_customer_admission: Mapping[str, int] | None = None,
        cache_exclusions: Mapping[str, bool] | None = None,
        cache_exclude_patterns: Sequence[str] | None = None,
        cache_customer_exclusions: Mapping[str, Mapping[str, Any]] | None = None,
        journal: bool | Journal | None = None,
        journal_path: str | os.PathLike[str] | None = None,
        journal_retention_days: float | None = None,
        avoided_call_cost: float | None = None,
        customer_avoided_call_cost: Mapping[str, float] | None = None,
        cache_customer_ttl: Mapping[str, float] | None = None,
        judge_call_cost: float | None = None,
        embedding_call_cost: float | None = None,
        **kwargs: Any,
    ) -> None:
        retry_policy = coerce_policy(retry)
        client = anthropic.Anthropic(*args, **sdk_kwargs(retry_policy, kwargs))
        state = build_state(
            caelum_key=caelum_key,
            cache_admission=cache_admission,
            cache_customer_admission=cache_customer_admission,
            cache_exclusions=cache_exclusions,
            cache_exclude_patterns=cache_exclude_patterns,
            cache_customer_exclusions=cache_customer_exclusions,
            journal=journal,
            journal_path=journal_path,
            journal_retention_days=journal_retention_days,
            avoided_call_cost=avoided_call_cost,
            customer_avoided_call_cost=customer_avoided_call_cost,
            cache_customer_ttl=cache_customer_ttl,
            judge_call_cost=judge_call_cost,
            embedding_call_cost=embedding_call_cost,
            cache=cache,
            cache_ttl=cache_ttl,
            cache_path=cache_path,
            cache_store=cache_store,
            retry_policy=retry_policy,
            semantic_cache=semantic_cache,
            semantic_threshold=semantic_threshold,
            semantic_gray_threshold=semantic_gray_threshold,
            semantic_customer_thresholds=semantic_customer_thresholds,
            semantic_threshold_floor=semantic_threshold_floor,
            semantic_embedder=semantic_embedder,
            semantic_verifier=semantic_verifier,
            router_config=router_config,
            fallback=fallback,
            provider_keys=provider_keys,
            provider_clients=provider_clients,
            native_provider=_PROVIDER,
            is_async=False,
        )
        _bind(self, client, state, is_async=False)

    def _caelum_rewrap(self, client: Any) -> Any:
        new = object.__new__(type(self))
        _bind(new, client, getattr(self, _STATE_ATTR), is_async=False)
        return new

    @property
    def metrics(self) -> Metrics:
        """Local metrics for this client: requests, cache hits, costs, latency."""
        state: CaelumState = getattr(self, _STATE_ATTR)
        return state.metrics

    @property
    def journal(self) -> Journal | None:
        """Per-request journal of this client (None when no cache is enabled)."""
        state: CaelumState = getattr(self, _STATE_ATTR)
        return state.journal

    def clear_cache(
        self, customer: str = "", older_than: float | None = None
    ) -> dict[str, int | None]:
        """Delete one end customer's cached answers, locally and in CAELUM Cloud.

        ``customer``: the name used with ``caelum.customer()`` (``""`` = requests made
        without one). Other customers are never touched. ``older_than``: only delete
        answers cached more than that many seconds ago. Returns the number deleted
        (``{"local": n, "cloud": n}``) and notes the clearing in the request journal.
        """
        state: CaelumState = getattr(self, _STATE_ATTR)
        return clear_customer_cache(state, customer, older_than)

    def shared_admission_pending(self, customer: str = "") -> int | None:
        """Questions of ``customer`` counted in CAELUM Cloud and still waiting for
        repetitions (shared admission). None without CAELUM Cloud or if unreachable."""
        state: CaelumState = getattr(self, _STATE_ATTR)
        if state.cloud is None:
            return None
        return state.cloud.admission_pending(customer_tag(customer))


class AsyncCaelumAnthropic(_AsyncBase):
    """Drop-in replacement for ``anthropic.AsyncAnthropic``.

    Accepts exactly the same arguments as ``anthropic.AsyncAnthropic``, plus the same
    optional CAELUM options as :class:`CaelumAnthropic`.
    """

    def __init__(
        self,
        *args: Any,
        cache: bool = False,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        cache_path: str | os.PathLike[str] | None = None,
        cache_store: CacheStore | None = None,
        retry: bool | RetryPolicy | None = None,
        semantic_cache: bool = False,
        semantic_threshold: float | None = None,
        semantic_gray_threshold: float | None = None,
        semantic_customer_thresholds: Mapping[str, tuple[float, float]] | None = None,
        semantic_threshold_floor: float | None = None,
        semantic_embedder: Embedder | AsyncEmbedder | None = None,
        semantic_verifier: Verifier | AsyncVerifier | bool | None = None,
        router_config: RouterConfig | Mapping[str, Any] | str | os.PathLike[str] | None = None,
        fallback: Sequence[str | Mapping[str, str]] | FallbackChain | None = None,
        provider_keys: Mapping[str, str] | None = None,
        provider_clients: Mapping[str, Any] | None = None,
        caelum_key: str | None = None,
        cache_admission: int | None = None,
        cache_customer_admission: Mapping[str, int] | None = None,
        cache_exclusions: Mapping[str, bool] | None = None,
        cache_exclude_patterns: Sequence[str] | None = None,
        cache_customer_exclusions: Mapping[str, Mapping[str, Any]] | None = None,
        journal: bool | Journal | None = None,
        journal_path: str | os.PathLike[str] | None = None,
        journal_retention_days: float | None = None,
        avoided_call_cost: float | None = None,
        customer_avoided_call_cost: Mapping[str, float] | None = None,
        cache_customer_ttl: Mapping[str, float] | None = None,
        judge_call_cost: float | None = None,
        embedding_call_cost: float | None = None,
        **kwargs: Any,
    ) -> None:
        retry_policy = coerce_policy(retry)
        client = anthropic.AsyncAnthropic(*args, **sdk_kwargs(retry_policy, kwargs))
        state = build_state(
            caelum_key=caelum_key,
            cache_admission=cache_admission,
            cache_customer_admission=cache_customer_admission,
            cache_exclusions=cache_exclusions,
            cache_exclude_patterns=cache_exclude_patterns,
            cache_customer_exclusions=cache_customer_exclusions,
            journal=journal,
            journal_path=journal_path,
            journal_retention_days=journal_retention_days,
            avoided_call_cost=avoided_call_cost,
            customer_avoided_call_cost=customer_avoided_call_cost,
            cache_customer_ttl=cache_customer_ttl,
            judge_call_cost=judge_call_cost,
            embedding_call_cost=embedding_call_cost,
            cache=cache,
            cache_ttl=cache_ttl,
            cache_path=cache_path,
            cache_store=cache_store,
            retry_policy=retry_policy,
            semantic_cache=semantic_cache,
            semantic_threshold=semantic_threshold,
            semantic_gray_threshold=semantic_gray_threshold,
            semantic_customer_thresholds=semantic_customer_thresholds,
            semantic_threshold_floor=semantic_threshold_floor,
            semantic_embedder=semantic_embedder,
            semantic_verifier=semantic_verifier,
            router_config=router_config,
            fallback=fallback,
            provider_keys=provider_keys,
            provider_clients=provider_clients,
            native_provider=_PROVIDER,
            is_async=True,
        )
        _bind(self, client, state, is_async=True)

    def _caelum_rewrap(self, client: Any) -> Any:
        new = object.__new__(type(self))
        _bind(new, client, getattr(self, _STATE_ATTR), is_async=True)
        return new

    @property
    def metrics(self) -> Metrics:
        """Local metrics for this client: requests, cache hits, costs, latency."""
        state: CaelumState = getattr(self, _STATE_ATTR)
        return state.metrics

    @property
    def journal(self) -> Journal | None:
        """Per-request journal of this client (None when no cache is enabled)."""
        state: CaelumState = getattr(self, _STATE_ATTR)
        return state.journal

    def clear_cache(
        self, customer: str = "", older_than: float | None = None
    ) -> dict[str, int | None]:
        """Delete one end customer's cached answers, locally and in CAELUM Cloud.

        ``customer``: the name used with ``caelum.customer()`` (``""`` = requests made
        without one). Other customers are never touched. ``older_than``: only delete
        answers cached more than that many seconds ago. Returns the number deleted
        (``{"local": n, "cloud": n}``) and notes the clearing in the request journal.
        """
        state: CaelumState = getattr(self, _STATE_ATTR)
        return clear_customer_cache(state, customer, older_than)

    def shared_admission_pending(self, customer: str = "") -> int | None:
        """Questions of ``customer`` counted in CAELUM Cloud and still waiting for
        repetitions (shared admission). None without CAELUM Cloud or if unreachable."""
        state: CaelumState = getattr(self, _STATE_ATTR)
        if state.cloud is None:
            return None
        return state.cloud.admission_pending(customer_tag(customer))
