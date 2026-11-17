"""Drop-in wrapper for the Mistral SDK (``mistralai``).

Usage::

    from caelum import Mistral  # instead of: from mistralai.client import Mistral

    client = Mistral(api_key="...", cache=True)
    client.chat.complete(model="mistral-small-latest", messages=[...])
    await client.chat.complete_async(model="mistral-small-latest", messages=[...])

Every constructor argument, method and attribute of ``mistralai.client.Mistral`` keeps
working. CAELUM intercepts ``chat.complete`` and ``chat.complete_async``; everything else
(``chat.stream``, ``chat.parse``, ``embeddings``, ``agents``...) is forwarded untouched.
Not supported yet for Mistral: the semantic cache, routing and cross-provider fallback.
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
from caelum._proxy import ClientProxy, ResourceProxy
from caelum.cache.stores import CacheStore
from caelum.cloud import customer_tag
from caelum.journal import Journal
from caelum.metrics import Metrics
from caelum.retry import RetryPolicy, coerce_policy
from caelum.utils import credential_scope, missing_extra_message

try:
    from mistralai.client import Mistral as _Mistral
    from mistralai.client.models import ChatCompletionResponse
except ImportError as exc:  # pragma: no cover - exercised via a dedicated test
    raise ImportError(missing_extra_message("mistral", "caelum.mistral")) from exc

if TYPE_CHECKING:
    # WHY: type checkers see the real SDK class, so users keep full typing.
    _Base = _Mistral
else:
    _Base = ClientProxy

__all__ = ["CaelumMistral"]

_PROVIDER = "mistral"
_STATE_ATTR = "_caelum_state"
CHAT_COMPLETE = Endpoint(
    provider=_PROVIDER, name="chat.complete", response_type=ChatCompletionResponse
)


class _Chat(ResourceProxy):
    """``client.chat`` with ``complete`` / ``complete_async`` going through CAELUM."""

    def __init__(self, target: Any, engine: Engine) -> None:
        super().__init__(target)
        self._caelum_engine = engine

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        """The SDK ``complete`` method, routed through the CAELUM pipeline."""
        target = self._caelum_target
        if args:  # the SDK only accepts keyword arguments: let it raise as usual.
            return target.complete(*args, **kwargs)
        return self._caelum_engine.execute(
            CHAT_COMPLETE, kwargs, lambda request: target.complete(**request)
        )

    async def complete_async(self, *args: Any, **kwargs: Any) -> Any:
        """The SDK ``complete_async`` method, routed through the CAELUM pipeline."""
        target = self._caelum_target
        if args:
            return await target.complete_async(*args, **kwargs)
        return await self._caelum_engine.aexecute(
            CHAT_COMPLETE, kwargs, lambda request: target.complete_async(**request)
        )


def _namespace(client: Any) -> str:
    """Identity of the upstream server, so different endpoints never share answers."""
    try:
        server_url = str(client.sdk_configuration.get_server_details()[0])
    except Exception:  # pragma: no cover - SDK internals changed
        server_url = ""
    try:
        secret = client.sdk_configuration.security.api_key
    except AttributeError:  # pragma: no cover - SDK internals changed
        secret = None
    return f"mistral|{server_url}{credential_scope(secret)}"


def _bind(wrapper: object, client: Any, state: CaelumState) -> None:
    engine = Engine(namespace=lambda: _namespace(client), state=state)
    object.__setattr__(wrapper, "_caelum_client", client)
    object.__setattr__(wrapper, _STATE_ATTR, state)
    object.__setattr__(wrapper, "_caelum_engine", engine)
    object.__setattr__(wrapper, "_caelum_resources", {"chat": _Chat(client.chat, engine)})


class CaelumMistral(_Base):
    """Drop-in replacement for ``mistralai.client.Mistral``.

    Accepts exactly the same arguments as ``mistralai.client.Mistral``, plus the optional
    CAELUM options below. ``chat.complete`` and ``chat.complete_async`` share the same
    cache and metrics.

    Args:
        cache: Enable the exact response cache. Disabled by default: without it, CAELUM
            behaves exactly like the Mistral SDK.
        cache_ttl: Cache entry lifetime in seconds (default 24h, None = no expiry).
        cache_path: SQLite cache file (default ``~/.caelum/cache.sqlite``, or
            ``$CAELUM_HOME/cache.sqlite``).
        cache_store: Custom :class:`~caelum.cache.CacheStore` (advanced).
        retry: Enable CAELUM retries (``True`` or a :class:`~caelum.RetryPolicy`).
            Disabled by default. Do not combine with the SDK's own ``retry_config``, or
            attempts multiply.
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
        **kwargs: Forwarded to ``mistralai.client.Mistral``.
    """

    def __init__(
        self,
        *args: Any,
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
        client = _Mistral(*args, **kwargs)
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

    def __enter__(self) -> Any:
        self._caelum_client.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._caelum_client.__exit__(*exc_info)

    async def __aenter__(self) -> Any:
        await self._caelum_client.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._caelum_client.__aexit__(*exc_info)

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
