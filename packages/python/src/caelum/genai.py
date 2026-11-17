"""Drop-in wrapper for the Google Gen AI SDK (Gemini).

Usage::

    from caelum import genai  # instead of: from google import genai

    client = genai.Client(api_key="...", cache=True)
    client.models.generate_content(model="gemini-2.5-flash", contents="Hello")
    await client.aio.models.generate_content(model="gemini-2.5-flash", contents="Hello")

Everything else in this module (``genai.types``, ``genai.errors``...) is the original
``google.genai`` module, so only the import line changes.

CAELUM intercepts ``models.generate_content`` (sync and ``aio``). Every other method
(``generate_content_stream``, ``chats``, ``files``, ``embed_content``...) is forwarded
untouched. Not supported yet for Gemini: the semantic cache, routing and cross-provider
fallback (OpenAI/Anthropic requests cannot be translated to Gemini yet).
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
)
from caelum._proxy import ResourceProxy, SyncClientProxy
from caelum.cache.stores import CacheStore
from caelum.cloud import customer_tag
from caelum.journal import Journal
from caelum.metrics import Metrics
from caelum.retry import RetryPolicy, coerce_policy
from caelum.utils import credential_scope, missing_extra_message

try:
    from google import genai as _genai
    from google.genai import errors, types
except ImportError as exc:  # pragma: no cover - exercised via a dedicated test
    raise ImportError(missing_extra_message("gemini", "caelum.genai")) from exc

if TYPE_CHECKING:
    # WHY: type checkers see the real SDK class, so users keep full typing.
    _Base = _genai.Client
else:
    _Base = SyncClientProxy

__all__ = ["Client", "errors", "types"]

_PROVIDER = "gemini"
_STATE_ATTR = "_caelum_state"
GENERATE_CONTENT = Endpoint(
    provider=_PROVIDER,
    name="models.generate_content",
    response_type=types.GenerateContentResponse,
)


class _Models(ResourceProxy):
    """``client.models`` with ``generate_content`` going through the CAELUM engine."""

    def __init__(self, target: Any, engine: Engine) -> None:
        super().__init__(target)
        self._caelum_engine = engine

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """The SDK ``generate_content`` method, routed through the CAELUM pipeline."""
        target = self._caelum_target
        if args:  # the SDK only accepts keyword arguments: let it raise as usual.
            return target.generate_content(*args, **kwargs)
        return self._caelum_engine.execute(
            GENERATE_CONTENT, kwargs, lambda request: target.generate_content(**request)
        )


class _AsyncModels(ResourceProxy):
    """``client.aio.models`` with ``generate_content`` going through the CAELUM engine."""

    def __init__(self, target: Any, engine: Engine) -> None:
        super().__init__(target)
        self._caelum_engine = engine

    async def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """The SDK async ``generate_content`` method, routed through the CAELUM pipeline."""
        target = self._caelum_target
        if args:
            return await target.generate_content(*args, **kwargs)
        return await self._caelum_engine.aexecute(
            GENERATE_CONTENT, kwargs, lambda request: target.generate_content(**request)
        )


class _Aio(ResourceProxy):
    """``client.aio``: the async client, with ``models`` intercepted."""

    def __init__(self, target: Any, engine: Engine) -> None:
        super().__init__(target)
        self.models = _AsyncModels(target.models, engine)


def _namespace(client: Any) -> str:
    """Identity of the upstream server, so different backends never share answers."""
    backend = "vertex" if getattr(client, "vertexai", False) else "developer"
    try:
        base_url = str(client._api_client._http_options.base_url)  # noqa: SLF001
    except AttributeError:  # pragma: no cover - SDK internals changed
        base_url = ""
    api = getattr(client, "_api_client", None)
    scope = credential_scope(
        getattr(api, "api_key", None), getattr(api, "project", None), getattr(api, "location", None)
    )
    return f"gemini|{backend}|{base_url}{scope}"


def _bind(wrapper: object, client: Any, state: CaelumState) -> None:
    engine = Engine(namespace=lambda: _namespace(client), state=state)
    resources: dict[str, Any] = {
        "models": _Models(client.models, engine),
        "aio": _Aio(client.aio, engine),
    }
    object.__setattr__(wrapper, "_caelum_client", client)
    object.__setattr__(wrapper, _STATE_ATTR, state)
    object.__setattr__(wrapper, "_caelum_engine", engine)
    object.__setattr__(wrapper, "_caelum_resources", resources)


class Client(_Base):
    """Drop-in replacement for ``google.genai.Client``.

    Accepts exactly the same arguments as ``google.genai.Client``, plus the optional
    CAELUM options below. ``client.aio`` shares the same cache and metrics.

    Args:
        cache: Enable the exact response cache. Disabled by default: without it, CAELUM
            behaves exactly like the Google SDK.
        cache_ttl: Cache entry lifetime in seconds (default 24h, None = no expiry).
        cache_path: SQLite cache file (default ``~/.caelum/cache.sqlite``, or
            ``$CAELUM_HOME/cache.sqlite``).
        cache_store: Custom :class:`~caelum.cache.CacheStore` (advanced).
        retry: Enable CAELUM retries (``True`` or a :class:`~caelum.RetryPolicy`).
            Disabled by default. Do not combine with the SDK's own
            ``http_options.retry_options``, or attempts multiply.
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
        cache_customer_ttl: Lifetime of cached answers per end customer, in seconds
            (1 minute to 30 days), e.g. ``{"acme": 3600}``. A shorter lifetime
            also applies to answers already cached.
        caelum_key: CAELUM Cloud key (``sk-caelum-...``, default ``$CAELUM_API_KEY``).
            Sends usage metrics (never prompts) to your CAELUM Cloud dashboard and, with
            ``cache=True``, adds the shared Cloud cache behind the local one. Your AI calls
            still go directly to the provider. None (default) = no Cloud at all.
        **kwargs: Forwarded to ``google.genai.Client``.

    Requests whose ``tools`` contain Python functions (automatic function calling) are
    never cached, so your functions always run.
    """

    def __init__(
        self,
        *,
        cache: bool = False,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        cache_path: str | os.PathLike[str] | None = None,
        cache_store: CacheStore | None = None,
        retry: bool | RetryPolicy | None = None,
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
        **kwargs: Any,
    ) -> None:
        client = _genai.Client(**kwargs)
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
            cache=cache,
            cache_ttl=cache_ttl,
            cache_path=cache_path,
            cache_store=cache_store,
            retry_policy=coerce_policy(retry),
            native_provider=_PROVIDER,
        )
        _bind(self, client, state)

    def _caelum_rewrap(self, client: Any) -> Any:  # pragma: no cover - no copy() in SDK
        new = object.__new__(type(self))
        _bind(new, client, getattr(self, _STATE_ATTR))
        return new

    @property
    def metrics(self) -> Metrics:
        """Local metrics for this client (and ``client.aio``): requests, cache hits, costs."""
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


def __getattr__(name: str) -> Any:
    # WHY: anything else (genai.live, genai.Client types...) comes from google.genai.
    return getattr(_genai, name)
