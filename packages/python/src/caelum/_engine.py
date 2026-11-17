"""Provider-agnostic request pipeline shared by every CAELUM wrapper.

Each wrapped ``create`` call goes through :class:`Engine`::

    exact cache -> semantic cache -> API call (with retries) -> store in caches -> metrics

Wrappers stay thin: they only describe their endpoints with :class:`Endpoint`.

Robustness rule: CAELUM must never break a request that would have worked without it.
Any failure of a CAELUM feature (corrupted cache file, embedding API down, disk full...)
is logged and the request falls back to a plain API call.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
import os
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from caelum.cache import admission as admission_module
from caelum.cache.admission import AdmissionFilter
from caelum.cache.exact import ExactCache, compute_cache_key, should_cache
from caelum.cache.semantic import (
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_VERIFIED_THRESHOLD,
    AsyncEmbedder,
    Embedder,
    EmbeddingResult,
    SemanticCache,
    SemanticLookup,
    SemanticQuery,
    SemanticZones,
    check_floor,
    request_zones,
)
from caelum.cache.stores import CacheStore, SQLiteStore
from caelum.cache.ttl import TtlPolicy
from caelum.cache.vectors import SQLiteVectorStore, normalize
from caelum.cache.verifier import AsyncVerifier, Verdict, Verifier
from caelum.cloud import (
    DEFAULT_CLOUD_URL,
    CloudConnection,
    TieredStore,
    current_customer,
    customer_tag,
    resolve_cloud_key,
)
from caelum.context import (
    cache_bypassed,
    context_fingerprint,
    depends_on_conversation,
    last_user_text,
)
from caelum.exclusions import ExclusionPolicy
from caelum.fallback import FallbackChain, FallbackTier, ProviderPool, should_fallback
from caelum.journal import Journal, JournalEntry
from caelum.metrics import CacheKind, Metrics
from caelum.pricing import estimate_cost
from caelum.providers import dialect, is_local
from caelum.retry import RetryPolicy, async_with_retry, with_retry
from caelum.router.config import RouterConfig
from caelum.router.engine import RouteDecision, RouterEngine
from caelum.translate import UntranslatableRequestError, translate_request, translate_response
from caelum.utils import (
    annotate_response,
    annotate_route,
    extract_usage,
    logger,
    served_model,
    stable_hash,
)

__all__ = [
    "DEFAULT_CACHE_TTL",
    "CaelumState",
    "Endpoint",
    "Engine",
    "build_state",
    "sdk_kwargs",
]

SyncCall = Callable[[dict[str, Any]], Any]
AsyncCall = Callable[[dict[str, Any]], Awaitable[Any]]
QueryExtractor = Callable[[Mapping[str, Any], str], SemanticQuery | None]

DEFAULT_CACHE_TTL: float = 24 * 60 * 60  # 24 hours


@dataclass(frozen=True)
class Endpoint:
    """Static description of a wrapped SDK endpoint.

    Attributes:
        provider: Provider name, e.g. ``"openai"``.
        name: Endpoint name, e.g. ``"chat.completions"``.
        response_type: Pydantic model class of the (non-streaming) response, used to
            rebuild real SDK objects from cached payloads.
        semantic_query: Extracts the semantic-cache query from a request. None means the
            endpoint is never semantically cached (e.g. embeddings).
        routable: True for chat endpoints that support routing and fallback.
    """

    provider: str
    name: str
    response_type: type[Any]
    semantic_query: QueryExtractor | None = None
    routable: bool = False


@dataclass
class CaelumState:
    """CAELUM features configured for a client, shared by its copies."""

    exact_cache: ExactCache | None = None
    semantic_cache: SemanticCache | None = None
    metrics: Metrics = field(default_factory=Metrics)
    retry_policy: RetryPolicy | None = None
    router: RouterEngine | None = None
    fallback: FallbackChain | None = None
    providers: ProviderPool | None = None
    cloud: CloudConnection | None = None
    admission: AdmissionFilter | None = None
    exclusions: ExclusionPolicy | None = None
    journal: Journal | None = None
    ttl: TtlPolicy = field(default_factory=TtlPolicy)


def build_state(
    *,
    cache: bool,
    cache_ttl: float | None,
    cache_path: str | os.PathLike[str] | None,
    cache_store: CacheStore | None,
    retry_policy: RetryPolicy | None = None,
    semantic_cache: bool = False,
    semantic_threshold: float | None = None,
    semantic_gray_threshold: float | None = None,
    semantic_customer_thresholds: Mapping[str, tuple[float, float]] | None = None,
    semantic_threshold_floor: float | None = None,
    semantic_embedder: Embedder | AsyncEmbedder | None = None,
    default_embedder: Embedder | AsyncEmbedder | None = None,
    semantic_verifier: Verifier | AsyncVerifier | bool | None = None,
    default_verifier: Verifier | AsyncVerifier | None = None,
    router_config: Any = None,
    fallback: Any = None,
    provider_keys: Mapping[str, str] | None = None,
    provider_clients: Mapping[str, Any] | None = None,
    native_provider: str = "openai",
    is_async: bool = False,
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
) -> CaelumState:
    """Validate the CAELUM constructor options and build the shared state.

    Args:
        default_embedder: Embedder used when ``semantic_cache=True`` and no
            ``semantic_embedder`` is given (None = the provider has no default).
        semantic_verifier: Judge confirming semantic matches. None or True = the
            provider default (``default_verifier``); False = no verification.
        default_verifier: Verifier used when ``semantic_verifier`` is None/True.
        native_provider: Provider of the wrapped SDK client.
        is_async: Whether the wrapped client is asynchronous.
        caelum_key: CAELUM Cloud key (default ``$CAELUM_API_KEY``; None = no Cloud).
        cache_admission: API answers needed for a question before it is cached
            (default 2; 1 = cache at once).
        cache_customer_admission: Per end customer overrides of ``cache_admission``.
        cache_exclusions: Switches of the automatic exclusion categories (all on).
        cache_exclude_patterns: Extra regular expressions that exclude a question.
        cache_customer_exclusions: Per end customer exclusion settings.
        journal: Per-request journal: on by default when a cache is enabled; False to
            turn it off, or a :class:`caelum.journal.Journal` instance.
        journal_path: Journal file (default ``~/.caelum/journal.sqlite``).
        journal_retention_days: Days a journal line is kept (default 30).
        avoided_call_cost: Estimated cost of one avoided AI call when the real token
            cost is unknown (default $0.002).
        customer_avoided_call_cost: Per end customer ``avoided_call_cost``.
        cache_customer_ttl: Per end customer lifetime of cached answers, in seconds.
        judge_call_cost: Estimated cost of one judge call when its tokens are unknown.
        embedding_call_cost: Estimated cost of one embedding when its tokens are unknown.
    """
    router = RouterEngine(RouterConfig.load(router_config)) if router_config is not None else None
    chain = FallbackChain.parse(fallback) if fallback is not None else None
    providers = ProviderPool(
        native=native_provider,
        is_async=is_async,
        keys=provider_keys,
        clients=provider_clients,
        sdk_retries=retry_policy is None,
    )
    needed: frozenset[str] = frozenset()
    if router is not None:
        needed |= router.config.providers
    if chain is not None:
        needed |= chain.providers
    providers.require(needed)
    if cache_admission is not None and (
        isinstance(cache_admission, bool)
        or not isinstance(cache_admission, int)  # pyright: ignore[reportUnnecessaryIsInstance]
        or cache_admission < 1
    ):
        raise ValueError("cache_admission must be a whole number of at least 1")
    if cache_customer_admission is not None and not isinstance(
        cache_customer_admission,
        Mapping,  # pyright: ignore[reportUnnecessaryIsInstance]
    ):
        raise TypeError("cache_customer_admission must map a customer name to a number")
    for name, value in cast("Mapping[Any, Any]", cache_customer_admission or {}).items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                "cache_customer_admission must map a customer name to a whole number of at "
                "least 1, e.g. {'acme': 3}"
            )
    exclusions = ExclusionPolicy(
        cache_exclusions, cache_exclude_patterns, cache_customer_exclusions
    )
    ttl_policy = TtlPolicy(cache_ttl, cache_customer_ttl)
    check_floor(semantic_threshold_floor)  # refused before any cache file is opened
    cache_ttl = ttl_policy.default
    for name, flag in (("cache", cache), ("semantic_cache", semantic_cache)):
        if not isinstance(flag, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"{name} must be True or False")
    _check_options(
        cache=cache,
        semantic_cache=semantic_cache,
        cache_ttl=cache_ttl,
        cache_path=cache_path,
        cache_store=cache_store,
        semantic_threshold=semantic_threshold,
        semantic_gray_threshold=semantic_gray_threshold,
        semantic_customer_thresholds=semantic_customer_thresholds,
    )
    key = resolve_cloud_key(caelum_key)
    cloud = _cloud_connection(key) if key is not None else None
    exact_cache: ExactCache | None = None
    if cache:
        store = cache_store if cache_store is not None else _open_store(SQLiteStore, cache_path)
        if store is None and cloud is not None:
            # WHY: serverless platforms often have no writable home folder; with the Cloud
            # the shared cache still works, with a process-local memory in front of it.
            store = SQLiteStore(":memory:")
        if store is not None:
            if cloud is not None:
                store = TieredStore(store, cloud, cache_ttl)
            exact_cache = ExactCache(store, cache_ttl)
    semantic: SemanticCache | None = None
    if semantic_cache:
        embedder = semantic_embedder if semantic_embedder is not None else default_embedder
        if embedder is None:
            raise ValueError(
                "semantic_cache=True requires an embedder for this provider, e.g. "
                "semantic_embedder=caelum.OpenAIEmbedder(api_key=...). Note: the question "
                "text is then sent to OpenAI to compute its embedding."
            )
        verifier = _resolve_verifier(semantic_verifier, default_verifier)
        if semantic_threshold is None:
            threshold = DEFAULT_VERIFIED_THRESHOLD if verifier else DEFAULT_SEMANTIC_THRESHOLD
        else:
            threshold = semantic_threshold
        vectors = _open_store(SQLiteVectorStore, cache_path)
        if vectors is not None:
            semantic = SemanticCache(
                vectors,
                embedder,
                threshold=threshold,
                gray_threshold=semantic_gray_threshold,
                customer_thresholds=semantic_customer_thresholds,
                floor=semantic_threshold_floor,
                ttl=cache_ttl,
                verifier=verifier,
            )
    admission: AdmissionFilter | None = None
    if exact_cache is not None or semantic is not None:
        admission = AdmissionFilter(
            n=admission_module.DEFAULT_ADMISSION if cache_admission is None else cache_admission,
            customers=cache_customer_admission,
        )
    state = CaelumState(
        exact_cache=exact_cache,
        semantic_cache=semantic,
        admission=admission,
        exclusions=exclusions if admission is not None else None,
        ttl=ttl_policy,
        retry_policy=retry_policy,
        router=router,
        fallback=chain,
        providers=providers,
        cloud=cloud,
    )
    if admission is not None:
        state.metrics.attach_admission(admission)
        state.journal = _open_journal(
            journal,
            journal_path,
            journal_retention_days,
            avoided_call_cost,
            customer_avoided_call_cost,
            judge_call_cost,
            embedding_call_cost,
        )
        if state.journal is not None:
            state.metrics.attach_journal(state.journal)
    return state


def clear_customer_cache(
    state: CaelumState, customer: str, older_than: float | None = None
) -> dict[str, int | None]:
    """Delete one end customer's cached answers, locally and in CAELUM Cloud.

    Only this customer's entries are touched (``""`` = requests made outside any
    ``caelum.customer()`` block). With ``older_than`` (seconds), only entries written
    longer ago than that are deleted. The clearing is noted in the request journal.

    Returns ``{"local": n, "cloud": n}``; ``cloud`` is None without CAELUM Cloud or when
    it could not be reached.
    """
    if not isinstance(customer, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError('customer doit être un texte ("" pour les requêtes sans client)')
    if older_than is not None and (not _is_number(older_than) or older_than <= 0):
        raise ValueError(
            "older_than doit être une durée positive en secondes, par exemple 7 * 86400 "
            "pour une semaine [older_than must be a positive number of seconds]"
        )
    local = 0
    cloud: int | None = None
    exact = state.exact_cache
    if exact is not None:
        store: Any = exact.store
        if isinstance(store, TieredStore):
            if getattr(store.local, "supports_owner", False):
                local += cast(Any, store.local).clear_customer(customer, older_than)
            cloud = store.cloud.cache_clear(customer_tag(customer), older_than)
        elif getattr(store, "supports_owner", False):
            local += store.clear_customer(customer, older_than)
    semantic = state.semantic_cache
    if semantic is not None:
        local += semantic.store.delete_customer(customer, older_than)
    if state.journal is not None:
        state.journal.record(
            JournalEntry(
                outcome="purge",
                customer=customer,
                reason="clear" if older_than is None else f"older_than:{int(older_than)}s",
                entry_id=f"local:{local} cloud:{'-' if cloud is None else cloud}",
            )
        )
    return {"local": local, "cloud": cloud}


def _open_journal(
    journal: bool | Journal | None,
    path: str | os.PathLike[str] | None,
    retention_days: float | None,
    call_cost: float | None,
    customer_call_cost: Mapping[str, float] | None,
    judge_call_cost: float | None = None,
    embedding_call_cost: float | None = None,
) -> Journal | None:
    """The request journal, or None. A journal that cannot be opened never blocks a client."""
    if isinstance(journal, Journal):
        return journal
    if journal is False:
        return None
    if journal not in (None, True):
        raise TypeError("journal must be True, False or a caelum.journal.Journal")
    options: dict[str, Any] = {}
    if retention_days is not None:
        options["retention_days"] = retention_days
    if call_cost is not None:
        options["call_cost"] = call_cost
    if customer_call_cost is not None:
        options["customer_call_cost"] = customer_call_cost
    if judge_call_cost is not None:
        options["judge_call_cost"] = judge_call_cost
    if embedding_call_cost is not None:
        options["embedding_call_cost"] = embedding_call_cost
    try:
        return Journal(path, **options)
    except (ValueError, TypeError):
        raise  # wrong settings: say so at construction time
    except Exception:
        logger.warning(
            "caelum: the request journal could not be opened, continuing without it", exc_info=True
        )
        return None


_cloud_connections: dict[str, CloudConnection] = {}
_cloud_lock = threading.Lock()


def _cloud_connection(key: str) -> CloudConnection:
    """One connection (and one background thread) per CAELUM key and Cloud URL."""
    url = os.environ.get("CAELUM_CLOUD_URL") or DEFAULT_CLOUD_URL
    with _cloud_lock:
        connection = _cloud_connections.get(f"{url}|{key}")
        if connection is None:
            connection = CloudConnection(key, base_url=url)
            _cloud_connections[f"{url}|{key}"] = connection
            # WHY: the background thread opens a connection now, so the first shared-cache
            # lookup does not pay the TLS handshake.
            connection._start()  # pyright: ignore[reportPrivateUsage]
        return connection


def _is_number(value: object) -> bool:
    """A real, finite number (``True`` is not a number of seconds)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_options(
    *,
    cache: bool,
    semantic_cache: bool,
    cache_ttl: Any,
    cache_path: Any,
    cache_store: Any,
    semantic_threshold: Any,
    semantic_gray_threshold: Any = None,
    semantic_customer_thresholds: Any = None,
) -> None:
    """Reject wrong option values with a clear message, at construction time."""
    if semantic_threshold is not None and (
        not _is_number(semantic_threshold) or not 0 < semantic_threshold <= 1
    ):
        raise ValueError("semantic_threshold must be a number in the interval (0, 1]")
    if semantic_gray_threshold is not None:
        if not _is_number(semantic_gray_threshold) or not 0 < semantic_gray_threshold <= 1:
            raise ValueError("semantic_gray_threshold must be a number in the interval (0, 1]")
        if semantic_threshold is not None and semantic_gray_threshold > semantic_threshold:
            raise ValueError("semantic_gray_threshold must not be above semantic_threshold")
    if semantic_customer_thresholds is not None:
        if not isinstance(semantic_customer_thresholds, Mapping):
            raise TypeError(
                "semantic_customer_thresholds must map a customer name to "
                "(semantic_threshold, semantic_gray_threshold)"
            )
        for name, bounds in cast("Mapping[Any, Any]", semantic_customer_thresholds).items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(bounds, (tuple, list))
                or len(cast("tuple[Any, ...]", bounds)) != 2
                or not all(_is_number(b) for b in cast("tuple[Any, ...]", bounds))
            ):
                raise ValueError(
                    "semantic_customer_thresholds must map a customer name to "
                    "(semantic_threshold, semantic_gray_threshold), e.g. {'acme': (0.98, 0.93)}"
                )
    if cache_path is not None and (
        not isinstance(cache_path, (str, os.PathLike))  # pyright: ignore[reportUnknownArgumentType]
        or not str(cache_path).strip()  # pyright: ignore[reportUnknownArgumentType]
    ):
        raise TypeError("cache_path must be a non-empty file path (str or pathlib.Path)")
    if cache_store is not None and not isinstance(cache_store, CacheStore):
        raise TypeError("cache_store must be a caelum.cache.CacheStore instance")
    if not cache and not semantic_cache and (cache_path is not None or cache_store is not None):
        logger.warning("caelum: cache_path/cache_store given but caching is off; pass cache=True")


_StoreT = TypeVar("_StoreT")


def _open_store(
    factory: Callable[[str | os.PathLike[str] | None], _StoreT],
    path: str | os.PathLike[str] | None,
) -> _StoreT | None:
    """Open a cache file, or return None (cache disabled) if it cannot be used.

    The cache must never break the application: a read-only, corrupted or
    otherwise unusable file only disables caching, with a warning.
    """
    try:
        return factory(path)
    except (sqlite3.Error, OSError) as exc:
        logger.warning(
            "caelum: cache disabled, cannot use the cache file %s (%s); calling the API normally",
            path if path is not None else "~/.caelum/cache.sqlite",
            exc,
        )
        return None


def _resolve_verifier(
    requested: Verifier | AsyncVerifier | bool | None,
    default: Verifier | AsyncVerifier | None,
) -> Verifier | AsyncVerifier | None:
    if requested is False:
        return None
    if requested is None or requested is True:
        if requested is True and default is None:
            raise ValueError(
                "semantic_verifier=True requires a verifier for this provider, e.g. "
                "semantic_verifier=caelum.LLMVerifier(api_key=...)."
            )
        return default
    return requested


def sdk_kwargs(retry_policy: RetryPolicy | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Adjust the SDK constructor arguments for the enabled CAELUM features.

    When CAELUM retries are on, the SDK's own retries are disabled (unless the user set
    ``max_retries`` explicitly), so that attempts never multiply.
    """
    if retry_policy is not None and "max_retries" not in kwargs:
        return {**kwargs, "max_retries": 0}
    return kwargs


@dataclass
class _Plan:
    """What the pipeline knows about one request."""

    started: float
    provider: str = ""
    route: RouteDecision | None = None
    namespace: str = ""
    key_request: dict[str, Any] = field(default_factory=lambda: {})
    exact_key: str | None = None
    semantic_query: SemanticQuery | None = None
    embedding: EmbeddingResult | None = None
    semantic_replace: int | None = None  # gray zone: entry replaced by the fresh answer
    no_cache: bool = False  # conversation or cache_bypass: never read nor write a cache
    skip_reason: str | None = None  # "conversation", "cache_bypass" or an exclusion category
    excluded: bool = False
    semantic_score: float | None = None
    semantic_entry: str | None = None
    semantic_origin: str | None = None
    request_zones: SemanticZones | None = None  # bounds asked by this request
    threshold_used: float | None = None
    threshold_source: str | None = None
    # Cost and time of the checks made before serving a semantic answer (improvement A).
    embedded: bool = False
    embed_cost: float | None = None  # real cost from tokens, None = unknown
    embed_ms: float = 0.0
    judge_calls: int = 0
    judge_cost_real: float = 0.0  # judge calls whose real token cost is known
    judge_unknown_calls: int = 0  # judge calls whose cost must be estimated
    judge_ms: float = 0.0
    admission_counter: str | None = None  # "local" or "shared"


class Engine:
    """Runs requests through the CAELUM pipeline for one client."""

    def __init__(self, *, namespace: str | Callable[[], str], state: CaelumState) -> None:
        """Create an engine.

        Args:
            namespace: Identifies the upstream target (base URL + credential hash), so
                that clients pointing to different servers or using different keys never
                share cached answers. A callable is evaluated on every request, so that
                ``client.api_key = ...`` immediately switches to the new key's cache.
            state: CAELUM features configured for the client.
        """
        self._namespace = namespace
        self.state = state

    @property
    def namespace(self) -> str:
        """Current cache namespace (recomputed from the live client if callable)."""
        namespace = self._namespace
        return namespace if isinstance(namespace, str) else namespace()

    def _native(self, endpoint: Endpoint) -> str:
        """Provider actually behind the wrapped client (e.g. ``ollama`` for a localhost URL)."""
        pool = self.state.providers
        return pool.native if pool is not None else endpoint.provider

    # -- public entry points -------------------------------------------------

    def execute(self, endpoint: Endpoint, request: dict[str, Any], call: SyncCall) -> Any:
        """Execute a synchronous request through the pipeline."""
        bypass = _pop_bypass(request)
        asked = _pop_thresholds(request)
        plan = self._plan(endpoint, request, bypass)
        self._apply_request_thresholds(plan, asked)
        cached = self._exact_lookup(endpoint, plan.exact_key)
        if cached is not None:
            return self._hit(request, cached, plan, "exact")
        plan.semantic_query = self._semantic_query(endpoint, request, plan)
        if plan.semantic_query is not None:
            plan.embedding = self._embed(plan.semantic_query, plan)
            cached = self._semantic_lookup(endpoint, plan)
            if cached is not None:
                return self._hit(request, cached, plan, "semantic")
        provider, model = self._primary(endpoint, request, plan)
        try:
            response = self._attempt(endpoint, request, provider, model, call)
        except Exception as error:
            if not self._can_fall_back(endpoint, error):
                self._failed(request, plan)
                raise
            for tier in self._tiers(endpoint, request):
                try:
                    response = self._attempt(endpoint, request, tier.provider, tier.model, call)
                except Exception:
                    logger.warning("caelum: fallback tier %s failed", tier, exc_info=True)
                    continue
                return self._finish_fallback(request, response, plan, tier)
            self._failed(request, plan)
            raise error from None
        return self._finish(request, response, plan)

    async def aexecute(self, endpoint: Endpoint, request: dict[str, Any], call: AsyncCall) -> Any:
        """Execute an asynchronous request through the pipeline."""
        # WHY: SQLite lookups take well under a millisecond, so they run inline
        # instead of paying the overhead of a worker thread.
        bypass = _pop_bypass(request)
        asked = _pop_thresholds(request)
        plan = self._plan(endpoint, request, bypass)
        self._apply_request_thresholds(plan, asked)
        cached = self._exact_lookup(endpoint, plan.exact_key)
        if cached is not None:
            return self._hit(request, cached, plan, "exact")
        plan.semantic_query = self._semantic_query(endpoint, request, plan)
        if plan.semantic_query is not None:
            plan.embedding = await self._aembed(plan.semantic_query, plan)
            cached = await self._asemantic_lookup(endpoint, plan)
            if cached is not None:
                return self._hit(request, cached, plan, "semantic")
        provider, model = self._primary(endpoint, request, plan)
        try:
            response = await self._aattempt(endpoint, request, provider, model, call)
        except Exception as error:
            if not self._can_fall_back(endpoint, error):
                self._failed(request, plan)
                raise
            for tier in self._tiers(endpoint, request):
                try:
                    response = await self._aattempt(
                        endpoint, request, tier.provider, tier.model, call
                    )
                except Exception:
                    logger.warning("caelum: fallback tier %s failed", tier, exc_info=True)
                    continue
                return self._finish_fallback(request, response, plan, tier)
            self._failed(request, plan)
            raise error from None
        if plan.semantic_query is not None and plan.embedding is not None:
            # WHY: storing a vector may wait on the SQLite writer: keep the loop free.
            return await asyncio.to_thread(self._finish, request, response, plan)
        return self._finish(request, response, plan)

    # -- routing and dispatch ------------------------------------------------------

    def _apply_request_thresholds(self, plan: _Plan, asked: tuple[Any, Any, Any]) -> None:
        """Validate the thresholds asked by this request (raised to the client's floor)."""
        cache = self.state.semantic_cache
        floor = cache.floor if cache is not None else None
        resolved = request_zones(*asked, floor)
        if resolved is not None:
            plan.request_zones, plan.threshold_source = resolved
            if plan.threshold_source == "request_floor":
                logger.warning(
                    "caelum: semantic_threshold %s is below the floor %s: the floor is used",
                    asked[0],
                    floor,
                )

    def _plan(self, endpoint: Endpoint, request: dict[str, Any], bypass: bool = False) -> _Plan:
        plan = _Plan(started=time.perf_counter())
        # WHY: "and to Ouaga?" only means something after "Do you deliver to Bobo?": an
        # answer that depends on the conversation must never be reused nor stored.
        if bypass or cache_bypassed():
            plan.no_cache, plan.skip_reason = True, "cache_bypass"
        elif depends_on_conversation(request):
            plan.no_cache, plan.skip_reason = True, "conversation"
        exclusions = self.state.exclusions
        if not plan.no_cache and exclusions is not None:
            # WHY: an order number, a phone number, an amount or "tomorrow" makes the
            # answer personal or time-bound: never reuse it, never store the question.
            category = exclusions.check(last_user_text(request), current_customer())
            if category is not None:
                plan.no_cache, plan.skip_reason, plan.excluded = True, category, True
                self.state.metrics.record_exclusion(category)
        plan.route = self._route(endpoint, request)
        plan.provider = plan.route.provider if plan.route is not None else self._native(endpoint)
        if plan.route is not None:
            # WHY: cache entries are keyed by where the request is actually sent.
            plan.namespace = f"{self.namespace}|route:{plan.route.provider}"
            plan.key_request = {**request, "model": plan.route.model}
        else:
            plan.namespace = self.namespace
            plan.key_request = request
        if plan.no_cache:
            return plan
        # Same customer, language, model and system instructions, or no reuse at all.
        model = plan.key_request.get("model")
        plan.namespace += "|ctx:" + context_fingerprint(
            plan.key_request, model if isinstance(model, str) else None
        )
        plan.exact_key = self._exact_key(endpoint, plan.key_request, plan.namespace)
        return plan

    def _route(self, endpoint: Endpoint, request: dict[str, Any]) -> RouteDecision | None:
        router = self.state.router
        if router is None or not endpoint.routable:
            return None
        try:
            decision = router.select_route(request)
        except Exception:  # pragma: no cover - routing must never break a request
            logger.warning("caelum: routing failed, keeping the requested model", exc_info=True)
            return None
        if decision is None:
            return None
        native = self._native(endpoint)
        if decision.provider == native and decision.model == request.get("model"):
            return None
        if decision.provider != native:
            try:
                translate_request(
                    endpoint.provider, decision.provider, {**request, "model": decision.model}
                )
            except UntranslatableRequestError as reason:
                logger.info("caelum: route %r skipped: %s", decision.route, reason)
                return None
            except Exception:
                logger.warning("caelum: route %r skipped (translation failed)", decision.route)
                return None
        self.state.metrics.record_route()
        return decision

    def _primary(self, endpoint: Endpoint, request: dict[str, Any], plan: _Plan) -> tuple[str, Any]:
        if plan.route is not None:
            return plan.route.provider, plan.route.model
        return self._native(endpoint), request.get("model")

    def _foreign(
        self, endpoint: Endpoint, request: dict[str, Any], provider: str, model: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Raw client and translated request for a provider other than the wrapped one."""
        pool = self.state.providers
        assert pool is not None
        translated = translate_request(endpoint.provider, provider, {**request, "model": model})
        raw = pool.get(provider)
        is_anthropic = dialect(provider) == "anthropic"
        method = raw.messages.create if is_anthropic else raw.chat.completions.create
        return method, translated

    def _attempt(
        self, endpoint: Endpoint, request: dict[str, Any], provider: str, model: Any, call: SyncCall
    ) -> Any:
        """One logical attempt (with CAELUM retries if enabled) on ``provider`` / ``model``."""
        if provider == self._native(endpoint):
            target = request if model == request.get("model") else {**request, "model": model}

            def once() -> Any:
                return call(target)
        else:
            method, translated = self._foreign(endpoint, request, provider, model)

            def once() -> Any:
                return translate_response(provider, endpoint.provider, method(**translated))

        policy = self.state.retry_policy
        if policy is None:
            return once()
        return with_retry(once, policy, on_retry=self.state.metrics.record_retry)

    async def _aattempt(
        self,
        endpoint: Endpoint,
        request: dict[str, Any],
        provider: str,
        model: Any,
        call: AsyncCall,
    ) -> Any:
        """Asynchronous counterpart of :meth:`_attempt`."""
        if provider == self._native(endpoint):
            target = request if model == request.get("model") else {**request, "model": model}

            async def once() -> Any:
                return await call(target)
        else:
            method, translated = self._foreign(endpoint, request, provider, model)

            async def once() -> Any:
                return translate_response(provider, endpoint.provider, await method(**translated))

        policy = self.state.retry_policy
        if policy is None:
            return await once()
        return await async_with_retry(once, policy, on_retry=self.state.metrics.record_retry)

    def _can_fall_back(self, endpoint: Endpoint, error: Exception) -> bool:
        return self.state.fallback is not None and endpoint.routable and should_fallback(error)

    def _tiers(self, endpoint: Endpoint, request: dict[str, Any]) -> list[FallbackTier]:
        chain = self.state.fallback
        assert chain is not None
        tiers: list[FallbackTier] = []
        native = self._native(endpoint)
        for tier in chain.tiers:
            if tier.provider != native:
                if request.get("stream") is True and dialect(tier.provider) != dialect(native):
                    continue  # streams are not translated between API formats yet
                try:
                    translate_request(endpoint.provider, tier.provider, request)
                except UntranslatableRequestError as reason:
                    logger.info("caelum: fallback tier %s skipped: %s", tier, reason)
                    continue
                except Exception:
                    logger.warning("caelum: fallback tier %s skipped (translation failed)", tier)
                    continue
            tiers.append(tier)
        return tiers

    # -- outcomes ----------------------------------------------------------------

    def _finish_fallback(
        self, request: dict[str, Any], response: Any, plan: _Plan, tier: FallbackTier
    ) -> Any:
        model = _requested_model(request)
        logger.warning("caelum: primary request failed, answered by fallback tier %s", tier)
        self.state.metrics.record_fallback()
        if request.get("stream") is not True:
            # WHY: fallback answers are never cached, so the next request tries the
            # primary again once it has recovered.
            annotate_response(response, cache_hit=False, cache_type=None)
        annotate_route(
            response, routed_from=model, routed_to=tier.model, provider=tier.provider, route=None
        )
        is_stream = request.get("stream") is True
        cost = None if is_stream else _cost(tier.provider, response, tier.model)
        self.state.metrics.record_api_call(
            model=model,
            latency_ms=_elapsed_ms(plan.started),
            cost_usd=cost,
            cache_miss=False,
        )
        self._report(request, None if is_stream else response, plan, cache="miss", cost=cost)
        self._log(request, plan)
        return response

    def _finish(self, request: dict[str, Any], response: Any, plan: _Plan) -> Any:
        model = _requested_model(request)
        if plan.route is not None:
            annotate_route(
                response,
                routed_from=model,
                routed_to=plan.route.model,
                provider=plan.route.provider,
                route=plan.route.route,
            )
        if request.get("stream") is True:
            # Streams are returned untouched; latency = time to open the stream.
            self.state.metrics.record_api_call(
                model=model, latency_ms=_elapsed_ms(plan.started), cost_usd=None, cache_miss=False
            )
            self._report(request, None, plan, cache="miss", cost=None)
            self._log(request, plan)
            return response
        payload = _dump(response)
        if payload is not None:
            if self._admit(plan):
                self._store(plan, payload)
            else:
                self._shared_admission(plan, payload)
        annotate_response(response, cache_hit=False, cache_type=None)
        cost = _cost(plan.provider, response, model)
        self.state.metrics.record_api_call(
            model=model,
            latency_ms=_elapsed_ms(plan.started),
            cost_usd=cost,
            cache_miss=plan.exact_key is not None or plan.semantic_query is not None,
        )
        self._report(request, response, plan, cache="miss", cost=cost)
        self._log(request, plan)
        return response

    def _store(self, plan: _Plan, payload: dict[str, Any]) -> None:
        if plan.exact_key is not None:
            self._exact_store(plan.exact_key, payload)
        if plan.semantic_query is not None and plan.embedding is not None:
            self._semantic_store(
                plan.semantic_query, plan.embedding, payload, plan.semantic_replace
            )

    def _shared_admission(self, plan: _Plan, payload: dict[str, Any]) -> None:
        """Not repeated enough on this server: ask the account-wide counter (Starter).

        Runs after the answer was returned (background thread of the Cloud connection).
        Only hashes are sent: the cache key and a hash of the end customer.
        """
        cloud, admission = self.state.cloud, self.state.admission
        if cloud is None or admission is None:
            return
        if plan.exact_key is None and (plan.semantic_query is None or plan.embedding is None):
            return  # excluded, bypassed or not cacheable: never sent
        customer = current_customer() or ""
        n = admission.admission_for(customer or None)
        if n <= 1:
            return
        fingerprint = plan.exact_key or stable_hash(
            [
                plan.semantic_query.context_key if plan.semantic_query else "",
                plan.semantic_query.text if plan.semantic_query else "",
            ]
        )
        context = contextvars.copy_context()  # the store runs later, as this customer
        metrics = self.state.metrics

        def decided(admitted: bool | None) -> None:
            metrics.record_shared_admission(admitted)
            if admitted:
                context.run(self._store, plan, payload)

        if cloud.admission_observe(
            fingerprint, customer_tag(customer), n, self.state.ttl.ttl_for(customer), decided
        ):
            plan.admission_counter = "shared"

    def _admit(self, plan: _Plan) -> bool:
        """Whether this API answer may be cached now (frequency admission)."""
        if plan.exact_key is None and (plan.semantic_query is None or plan.embedding is None):
            return False  # nothing would be stored: do not count it either
        admission = self.state.admission
        if admission is None or plan.semantic_replace is not None:
            # A gray-zone refresh replaces an entry that was already admitted.
            return True
        plan.admission_counter = "local"
        customer = current_customer()
        group, vector, threshold = "", None, 1.0
        semantic = self.state.semantic_cache
        if semantic is not None and plan.semantic_query is not None and plan.embedding is not None:
            group = plan.semantic_query.context_key
            try:
                vector = list(normalize(plan.embedding.vector))
            except ValueError:
                vector = None
            threshold = semantic.zones_for(customer).serve
        key = plan.exact_key or stable_hash(
            [group, plan.semantic_query.text if plan.semantic_query else ""]
        )
        try:
            return admission.observe(key, customer, group=group, vector=vector, threshold=threshold)
        except Exception:  # pragma: no cover - admission must never break a request
            logger.warning("caelum: cache admission failed", exc_info=True)
            return False

    def _hit(self, request: dict[str, Any], response: Any, plan: _Plan, kind: CacheKind) -> Any:
        model = _requested_model(request)
        annotate_response(response, cache_hit=True, cache_type=kind)
        if kind == "semantic" and plan.exact_key is not None:
            # Next time this exact wording is asked, skip the embedding call too.
            payload = _dump(response)
            if payload is not None:
                self._exact_store(plan.exact_key, payload)
        saved = _cost(plan.provider, response, model)
        self.state.metrics.record_cache_hit(
            kind=kind,
            model=model,
            latency_ms=_elapsed_ms(plan.started),
            saved_usd=saved,
        )
        self._report(request, response, plan, cache=kind, saved=saved)
        self._log(request, plan, kind=kind, saved=saved)
        return response

    def _log(
        self,
        request: dict[str, Any],
        plan: _Plan,
        *,
        kind: CacheKind | None = None,
        saved: float | None = None,
    ) -> None:
        """One line in the request journal. Never raises, never blocks."""
        journal = self.state.journal
        if journal is None:
            return
        try:
            customer = current_customer() or ""
            entry = JournalEntry(outcome="miss", customer=customer)
            entry.latency_ms = round(_elapsed_ms(plan.started), 2)
            entry.reason = plan.skip_reason
            if plan.embedded:
                if plan.embed_cost is not None:
                    entry.embedding_cost_usd, entry.embedding_cost_source = (
                        plan.embed_cost,
                        "tokens",
                    )
                else:
                    entry.embedding_cost_usd = journal.embedding_call_cost
                    entry.embedding_cost_source = "estimate"
            if plan.judge_calls:
                entry.judge_calls = plan.judge_calls
                entry.judge_cost_usd = (
                    plan.judge_cost_real + plan.judge_unknown_calls * journal.judge_call_cost
                )
                if plan.judge_unknown_calls == 0:
                    entry.judge_cost_source = "tokens"
                elif plan.judge_cost_real == 0:
                    entry.judge_cost_source = "estimate"
                else:
                    entry.judge_cost_source = "mixed"
            entry.overhead_ms = round(plan.embed_ms + plan.judge_ms, 2)
            entry.admission_counter = plan.admission_counter
            if plan.excluded:
                entry.outcome = "excluded"  # category only: the text is never logged
            elif plan.no_cache:
                entry.outcome = "bypassed"
            else:
                question = last_user_text(request) or None
                entry.question = question
                entry.score = plan.semantic_score
                entry.entry_id = plan.semantic_entry
                entry.origin_question = plan.semantic_origin
                entry.threshold = plan.threshold_used
                if plan.threshold_used is not None:
                    entry.threshold_source = plan.threshold_source
                if kind is not None:
                    entry.outcome, entry.cache_kind = "hit", kind
                    if kind == "exact":
                        entry.score, entry.origin_question = None, question
                        entry.entry_id = (plan.exact_key or "")[:16] or None
                    if saved is not None:
                        entry.saved_usd, entry.saved_source = saved, "tokens"
                    else:
                        entry.saved_usd = journal.call_cost_for(customer)
                        entry.saved_source = "estimate"
                elif plan.semantic_replace is not None:
                    entry.outcome = "gray"
            journal.record(entry)
        except Exception:  # pragma: no cover - the journal must never break a request
            logger.debug("caelum: could not log the request", exc_info=True)

    def _failed(self, request: dict[str, Any], plan: _Plan) -> None:
        self.state.metrics.record_failure(
            model=_requested_model(request), latency_ms=_elapsed_ms(plan.started)
        )
        self._report(request, None, plan, cache="error")

    def _report(
        self,
        request: dict[str, Any],
        response: Any,
        plan: _Plan,
        *,
        cache: str,
        cost: float | None = None,
        saved: float | None = None,
    ) -> None:
        """Queue a usage event for CAELUM Cloud (never the prompt or the answer)."""
        cloud = self.state.cloud
        if cloud is None:
            return
        try:
            usage = extract_usage(response) if response is not None else None
            model = (served_model(response) if response is not None else None) or _requested_model(
                request
            )
            cloud.record(
                {
                    "ts": time.time(),
                    "model": model or "unknown",
                    "provider": plan.provider,
                    "input_tokens": usage[0] if usage else 0,
                    "output_tokens": usage[1] if usage else 0,
                    "cost_usd": cost or 0.0,
                    "saved_usd": saved or 0.0,
                    "cache": cache,
                    "latency_ms": round(_elapsed_ms(plan.started), 2),
                    "customer": current_customer() or "",
                }
            )
        except Exception:  # pragma: no cover - metrics must never break a request
            logger.debug("caelum: could not queue a cloud event", exc_info=True)

    # -- exact cache -------------------------------------------------------------

    def _exact_key(self, endpoint: Endpoint, request: dict[str, Any], namespace: str) -> str | None:
        if self.state.exact_cache is None or not should_cache(request):
            return None
        return compute_cache_key(
            request, provider=endpoint.provider, endpoint=endpoint.name, namespace=namespace
        )

    def _exact_lookup(self, endpoint: Endpoint, key: str | None) -> Any | None:
        cache = self.state.exact_cache
        if cache is None or key is None:
            return None
        customer = current_customer() or ""
        try:
            payload = cache.get(key, self.state.ttl.ttl_for(customer), customer)
        except Exception:
            logger.warning("caelum: cache read failed, calling the API", exc_info=True)
            return None
        if payload is None:
            return None
        response = _rebuild(endpoint, payload)
        if response is None:
            # WHY: a payload written by an older SDK version may no longer parse.
            logger.info("caelum: dropping unreadable cache entry %s", key[:12])
            try:
                cache.delete(key)
            except Exception:  # pragma: no cover - best effort
                logger.debug("caelum: could not delete cache entry", exc_info=True)
        return response

    def _exact_store(self, key: str, payload: dict[str, Any]) -> None:
        cache = self.state.exact_cache
        if cache is None:
            return
        customer = current_customer() or ""
        try:
            cache.set(key, payload, self.state.ttl.ttl_for(customer), customer)
        except Exception:
            logger.warning("caelum: cache write failed", exc_info=True)

    # -- semantic cache ------------------------------------------------------------

    def _semantic_query(
        self, endpoint: Endpoint, request: dict[str, Any], plan: _Plan
    ) -> SemanticQuery | None:
        if self.state.semantic_cache is None or endpoint.semantic_query is None or plan.no_cache:
            return None
        return endpoint.semantic_query(plan.key_request or request, plan.namespace)

    def _record_embedding(self, embedding: EmbeddingResult, plan: _Plan, started: float) -> None:
        cache = self.state.semantic_cache
        assert cache is not None
        cost = estimate_cost(cache.embedder.model, embedding.tokens, 0)
        self.state.metrics.record_embedding_cost(cost)
        plan.embedded = True
        plan.embed_ms += _elapsed_ms(started)
        plan.embed_cost = cost if embedding.tokens > 0 else None

    def _embed(self, query: SemanticQuery, plan: _Plan) -> EmbeddingResult | None:
        cache = self.state.semantic_cache
        assert cache is not None
        started = time.perf_counter()
        try:
            embedding = cache.embed(query.text)
        except Exception:
            logger.warning("caelum: embedding failed, skipping the semantic cache", exc_info=True)
            return None
        self._record_embedding(embedding, plan, started)
        return embedding

    async def _aembed(self, query: SemanticQuery, plan: _Plan) -> EmbeddingResult | None:
        cache = self.state.semantic_cache
        assert cache is not None
        started = time.perf_counter()
        try:
            embedding = await cache.aembed(query.text)
        except Exception:
            logger.warning("caelum: embedding failed, skipping the semantic cache", exc_info=True)
            return None
        self._record_embedding(embedding, plan, started)
        return embedding

    def _candidate(self, plan: _Plan) -> SemanticLookup | None:
        """Closest stored answer above the similarity threshold (not yet verified)."""
        cache = self.state.semantic_cache
        if cache is None or plan.semantic_query is None or plan.embedding is None:
            return None
        try:
            customer = current_customer()
            # Priority: the request's bounds (never below the floor), then the customer's,
            # then the defaults. cache_bypass and exclusions were applied before (no lookup).
            if plan.request_zones is not None:
                zones = plan.request_zones
            else:
                zones, plan.threshold_source = cache.zones_with_source(customer)
            plan.threshold_used = zones.serve
            lookup = cache.search(
                plan.semantic_query,
                plan.embedding,
                customer,
                self.state.ttl.ttl_for(customer),
                zones,
            )
        except Exception:
            logger.warning("caelum: semantic cache read failed", exc_info=True)
            return None
        if lookup.payload is None or lookup.zone is None:
            return None
        plan.semantic_score = lookup.score
        plan.semantic_entry = None if lookup.row_id is None else str(lookup.row_id)
        plan.semantic_origin = lookup.stored_query
        if lookup.zone == "refresh":
            # Gray zone: close but not sure enough. Call the API; its answer replaces this
            # entry (see _semantic_store) instead of adding a near-duplicate.
            logger.debug("caelum: semantic gray zone (similarity %.4f)", lookup.score or 0.0)
            plan.semantic_replace = lookup.row_id
            self.state.metrics.record_semantic_refresh()
            return None
        return lookup

    @staticmethod
    def _accept(endpoint: Endpoint, lookup: SemanticLookup) -> Any | None:
        assert lookup.payload is not None
        logger.debug("caelum: semantic hit (similarity %.4f)", lookup.score or 0.0)
        return _rebuild(endpoint, lookup.payload)

    def _record_verdict(
        self,
        verifier: Verifier | AsyncVerifier,
        verdict: Verdict,
        plan: _Plan,
        started: float,
    ) -> None:
        cost = estimate_cost(verifier.model, verdict.input_tokens, verdict.output_tokens)
        self.state.metrics.record_verification(accepted=verdict.accepted, cost_usd=cost)
        plan.judge_ms += _elapsed_ms(started)
        calls = max(int(verdict.calls), 1)
        plan.judge_calls += calls
        if cost is not None and verdict.input_tokens > 0:
            plan.judge_cost_real += cost
        else:
            plan.judge_unknown_calls += calls  # priced later with the configured estimate

    def _semantic_lookup(self, endpoint: Endpoint, plan: _Plan) -> Any | None:
        lookup = self._candidate(plan)
        if lookup is None:
            return None
        cache = self.state.semantic_cache
        assert cache is not None and plan.semantic_query is not None
        verifier = cache.verifier
        if verifier is None:
            return self._accept(endpoint, lookup)
        if lookup.stored_query is None or not isinstance(verifier, Verifier):
            return None  # nothing to compare with, or an async-only verifier
        started = time.perf_counter()
        try:
            verdict = verifier.verify(lookup.stored_query, plan.semantic_query.text)
        except Exception:
            logger.warning("caelum: semantic verification failed, calling the API", exc_info=True)
            plan.judge_ms += _elapsed_ms(started)
            return None
        self._record_verdict(verifier, verdict, plan, started)
        return self._accept(endpoint, lookup) if verdict.accepted else None

    async def _asemantic_lookup(self, endpoint: Endpoint, plan: _Plan) -> Any | None:
        # WHY: a brute-force vector search can take a few hundred ms on a large cache;
        # it runs in a worker thread so that it never blocks the event loop.
        lookup = await asyncio.to_thread(self._candidate, plan)
        if lookup is None:
            return None
        cache = self.state.semantic_cache
        assert cache is not None and plan.semantic_query is not None
        verifier = cache.verifier
        if verifier is None:
            return self._accept(endpoint, lookup)
        if lookup.stored_query is None:
            return None
        started = time.perf_counter()
        try:
            if isinstance(verifier, AsyncVerifier):
                verdict = await verifier.averify(lookup.stored_query, plan.semantic_query.text)
            else:
                verdict = await asyncio.to_thread(
                    verifier.verify, lookup.stored_query, plan.semantic_query.text
                )
        except Exception:
            logger.warning("caelum: semantic verification failed, calling the API", exc_info=True)
            plan.judge_ms += _elapsed_ms(started)
            return None
        self._record_verdict(verifier, verdict, plan, started)
        return self._accept(endpoint, lookup) if verdict.accepted else None

    def _semantic_store(
        self,
        query: SemanticQuery,
        embedding: EmbeddingResult,
        payload: dict[str, Any],
        replace_id: int | None = None,
    ) -> None:
        cache = self.state.semantic_cache
        if cache is None:
            return
        try:
            customer = current_customer() or ""
            cache.add(
                query, embedding, payload, replace_id, self.state.ttl.ttl_for(customer), customer
            )
        except Exception:
            logger.warning("caelum: semantic cache write failed", exc_info=True)


def _dump(response: Any) -> dict[str, Any] | None:
    """JSON-compatible payload of an SDK response (None if it cannot be serialized)."""
    model_dump: Callable[..., Any] | None = getattr(response, "model_dump", None)
    if model_dump is None:
        return None
    try:
        payload = model_dump(mode="json")
    except Exception:  # pragma: no cover - defensive
        return None
    return payload if isinstance(payload, dict) else None  # pyright: ignore[reportUnknownVariableType]


def _pop_thresholds(request: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Remove CAELUM's per-request threshold arguments (never sent to the provider)."""
    return (
        request.pop("semantic_threshold", None),
        request.pop("semantic_gray_threshold", None),
        request.pop("semantic_strict", None),
    )


def _pop_bypass(request: dict[str, Any]) -> bool:
    """Remove CAELUM's ``cache_bypass`` argument (never sent to the provider)."""
    value = request.pop("cache_bypass", False)
    if not isinstance(value, bool):
        raise TypeError("cache_bypass must be True or False")
    return value


def _rebuild(endpoint: Endpoint, payload: dict[str, Any]) -> Any | None:
    """Rebuild a real SDK response object from a cached payload (None if invalid)."""
    try:
        return endpoint.response_type.model_validate(payload)
    except Exception:
        return None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _requested_model(request: dict[str, Any]) -> str | None:
    model = request.get("model")
    return model if isinstance(model, str) else None


def _cost(provider: str, response: Any, requested_model: str | None) -> float | None:
    """Estimated USD cost of ``response`` served by ``provider`` (local backends are free)."""
    if is_local(provider):
        return 0.0
    return _estimate_response_cost(response, requested_model)


def _estimate_response_cost(response: Any, requested_model: str | None) -> float | None:
    """Estimated USD cost of the tokens in ``response`` (None if unknown)."""
    usage = extract_usage(response)
    model = served_model(response) or (
        requested_model.removeprefix("models/") if requested_model else None
    )
    if usage is None or model is None:
        return None
    return estimate_cost(model, usage[0], usage[1])
