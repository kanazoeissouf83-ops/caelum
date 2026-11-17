"""Semantic cache: reuse answers to questions phrased differently but meaning the same.

Opt-in only (``semantic_cache=True``). How it works:

1. The **last user message** is turned into an embedding (OpenAI
   ``text-embedding-3-small`` by default, about $0.02 per million tokens).
2. CAELUM looks for the closest stored question and sorts its cosine similarity into
   three zones:

   * **serve** (``score >= semantic_threshold``): the stored answer is reused;
   * **gray zone** (``semantic_gray_threshold <= score < semantic_threshold``): close but
     not sure enough. The API is called and the new answer **replaces** the stored entry,
     so the cache keeps the freshest answer instead of piling up near-duplicates;
   * **miss** (below ``semantic_gray_threshold``): the API is called and the answer is
     stored as a new entry.

   Both bounds can be set per end customer (``semantic_customer_thresholds``, or
   :meth:`SemanticCache.set_thresholds`), e.g. stricter for a customer whose questions
   look alike but need different answers.
3. With a verifier (the default with the OpenAI wrapper), a small model checks that both
   questions ask for exactly the same thing before the answer is reused (see
   :mod:`caelum.cache.verifier`); the threshold is then a loose 0.60. Without a verifier,
   the threshold is a strict 0.97, which reuses very few paraphrases.
4. Only questions asked in **exactly the same context** are compared: same model,
   system prompt, conversation history, tools and generation parameters. Only the
   wording of the last question may differ.

WHY only the last user message varies (and not "all user messages", as first planned):
two requests with different system prompts ("translate to French" / "translate to
Spanish") or different histories must never share an answer. Restricting similarity to
the final question keeps the false-positive risk as low as possible.

WHY the embeddings endpoint is excluded: returning the vector of a *similar* text instead
of the requested one would be a silent correctness bug.

Privacy: embeddings and answers are stored locally. Computing an embedding sends the
question text to the embedding provider (OpenAI). With the OpenAI wrapper this is the
provider you already use; with other providers you must pass an embedder explicitly.
"""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from caelum.cache.exact import EXCLUDED_PARAMS, beta_headers, should_cache
from caelum.cache.vectors import SQLiteVectorStore
from caelum.cache.verifier import AsyncVerifier, Verifier
from caelum.utils import UnhashableRequestError, stable_hash, to_canonical

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GRAY_MARGIN",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "DEFAULT_VERIFIED_THRESHOLD",
    "AsyncEmbedder",
    "Embedder",
    "EmbeddingResult",
    "OpenAIEmbedder",
    "SemanticCache",
    "SemanticQuery",
    "SemanticZones",
    "anthropic_semantic_query",
    "openai_semantic_query",
]

DEFAULT_SEMANTIC_THRESHOLD = 0.97  # without a verifier: embeddings alone must be sure
DEFAULT_VERIFIED_THRESHOLD = 0.60  # with a verifier: candidates only, the judge decides
# Default width of the gray zone, just below the serve threshold.
DEFAULT_GRAY_MARGIN = 0.05
# Bounds of a per-request threshold (``semantic_threshold=`` on one request) and of the
# safety floor; ``semantic_strict=True`` asks for the highest one.
REQUEST_MIN_THRESHOLD = 0.80
REQUEST_MAX_THRESHOLD = 0.999
STRICT_GRAY_THRESHOLD = 0.95


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def check_floor(value: Any) -> float | None:
    """A valid safety floor (``semantic_threshold_floor``), or ValueError in French."""
    if value is None:
        return None
    if not _number(value) or not REQUEST_MIN_THRESHOLD <= value <= REQUEST_MAX_THRESHOLD:
        raise ValueError(
            f"semantic_threshold_floor : valeur invalide ({value!r}). Donnez un nombre entre "
            f"{REQUEST_MIN_THRESHOLD} et {REQUEST_MAX_THRESHOLD}, par exemple 0.95. "
            "[semantic_threshold_floor must be a number between 0.80 and 0.999]"
        )
    return float(value)


def request_zones(
    threshold: Any, gray_threshold: Any, strict: Any, floor: float | None
) -> tuple[SemanticZones, str] | None:
    """Bounds asked by ONE request, and where they come from.

    Returns None when the request asks for nothing, else ``(zones, source)`` with source
    ``"request"``, ``"request_floor"`` (raised to the client's floor) or ``"strict"``.
    Invalid values raise ValueError with a French message.
    """
    if strict is not None and not isinstance(strict, bool):
        raise ValueError(
            "semantic_strict doit valoir True ou False [semantic_strict must be True or False]"
        )
    if strict:
        if threshold is not None or gray_threshold is not None:
            raise ValueError(
                "semantic_strict=True fixe déjà le seuil : ne donnez pas aussi "
                "semantic_threshold ou semantic_gray_threshold "
                "[semantic_strict cannot be combined with explicit thresholds]"
            )
        return SemanticZones(serve=REQUEST_MAX_THRESHOLD, refresh=STRICT_GRAY_THRESHOLD), "strict"
    if threshold is None:
        if gray_threshold is not None:
            raise ValueError(
                "semantic_gray_threshold sur une requête demande aussi semantic_threshold "
                "[a per-request semantic_gray_threshold needs semantic_threshold]"
            )
        return None
    if not _number(threshold) or not REQUEST_MIN_THRESHOLD <= threshold <= REQUEST_MAX_THRESHOLD:
        raise ValueError(
            f"semantic_threshold (requête) : valeur invalide ({threshold!r}). Donnez un "
            f"nombre entre {REQUEST_MIN_THRESHOLD} et {REQUEST_MAX_THRESHOLD}, par exemple "
            "0.98, ou semantic_strict=True pour le plus strict. "
            "[a per-request semantic_threshold must be between 0.80 and 0.999]"
        )
    if gray_threshold is not None and (
        not _number(gray_threshold) or not 0 < gray_threshold < threshold
    ):
        raise ValueError(
            f"semantic_gray_threshold (requête) : valeur invalide ({gray_threshold!r}). La "
            f"borne basse doit être un nombre positif et strictement inférieur au seuil "
            f"({threshold}). [semantic_gray_threshold must be below semantic_threshold]"
        )
    serve, source = float(threshold), "request"
    if floor is not None and serve < floor:
        # WHY: the client's floor protects against a too-loose threshold written by
        # mistake in the app: the request still runs, at the floor.
        serve, source = floor, "request_floor"
    refresh = float(gray_threshold) if gray_threshold is not None else serve - DEFAULT_GRAY_MARGIN
    if refresh >= serve:
        refresh = serve - DEFAULT_GRAY_MARGIN
    return SemanticZones(serve=serve, refresh=round(refresh, 6)), source


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
KEY_VERSION = 2  # 2: customer, language, model and instructions added (0.2)
# Embedding models accept ~8k tokens; longer questions are simply not semantically cached.
MAX_QUERY_CHARS = 20_000
_QUERY_PLACEHOLDER = "<caelum:semantic-query>"


@dataclass(frozen=True)
class EmbeddingResult:
    """An embedding vector and the number of tokens billed to compute it."""

    vector: list[float]
    tokens: int


@runtime_checkable
class Embedder(Protocol):
    """Turns text into an embedding vector (synchronous)."""

    @property
    def model(self) -> str:
        """Embedding model name (entries from different models are never compared)."""
        ...

    def embed(self, text: str) -> EmbeddingResult:
        """Return the embedding of ``text``."""
        ...


@runtime_checkable
class AsyncEmbedder(Protocol):
    """Embedder with a native asynchronous method."""

    @property
    def model(self) -> str:
        """Embedding model name."""
        ...

    async def aembed(self, text: str) -> EmbeddingResult:
        """Return the embedding of ``text``."""
        ...


class OpenAIEmbedder:
    """Embedder backed by the OpenAI embeddings API.

    Args:
        client: A raw ``openai.OpenAI`` client (sync use).
        async_client: A raw ``openai.AsyncOpenAI`` client (async use).
        api_key: Used to create clients when none is given (defaults to the
            ``OPENAI_API_KEY`` environment variable).
        model: Embedding model.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        async_client: Any = None,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._client = client
        self._async_client = async_client
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> EmbeddingResult:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key)
        response = self._client.embeddings.create(model=self._model, input=text)
        return _to_result(response)

    async def aembed(self, text: str) -> EmbeddingResult:
        if self._async_client is None:
            import openai

            self._async_client = openai.AsyncOpenAI(api_key=self._api_key)
        response = await self._async_client.embeddings.create(model=self._model, input=text)
        return _to_result(response)


def _to_result(response: Any) -> EmbeddingResult:
    vector = [float(x) for x in response.data[0].embedding]
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    return EmbeddingResult(vector=vector, tokens=int(tokens or 0))


# ---------------------------------------------------------------------------
# Extracting the query and its context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticQuery:
    """The varying part of a request (``text``) and everything that must match exactly."""

    text: str
    context_key: str


def _text_only(content: Any) -> str | None:
    """Return the text of a message content, or None if it contains non-text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in cast("list[Any]", content):
            if not isinstance(part, Mapping):
                return None
            mapping = cast("Mapping[str, Any]", part)
            if mapping.get("type") != "text" or not isinstance(mapping.get("text"), str):
                return None  # images, audio, files, tool results...
            texts.append(cast(str, mapping["text"]))
        return "\n".join(texts) if texts else None
    return None


def _build_query(
    request: Mapping[str, Any], *, provider: str, endpoint: str, namespace: str
) -> SemanticQuery | None:
    if not should_cache(request):
        return None
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    message_list = cast("list[Any]", messages)
    last = message_list[-1]
    if not isinstance(last, Mapping):
        return None
    last_message = cast("Mapping[str, Any]", last)
    if last_message.get("role") != "user":
        return None
    text = _text_only(last_message.get("content"))
    if text is None or not text.strip() or len(text) > MAX_QUERY_CHARS:
        return None
    context = {
        name: value
        for name, value in request.items()
        if name not in EXCLUDED_PARAMS and not (name == "stream" and value is False)
    }
    betas = beta_headers(request.get("extra_headers"))
    if betas:
        context["__beta_headers__"] = betas
    templated = copy.deepcopy(message_list)
    templated[-1] = {**last_message, "content": _QUERY_PLACEHOLDER}
    context["messages"] = templated
    try:
        canonical = to_canonical(context)
    except UnhashableRequestError:
        return None
    context_key = stable_hash(
        {
            "v": KEY_VERSION,
            "provider": provider,
            "endpoint": endpoint,
            "namespace": namespace,
            "context": canonical,
        }
    )
    return SemanticQuery(text=text, context_key=context_key)


def openai_semantic_query(request: Mapping[str, Any], namespace: str) -> SemanticQuery | None:
    """Semantic query of an OpenAI ``chat.completions.create`` request (or None)."""
    return _build_query(
        request, provider="openai", endpoint="chat.completions", namespace=namespace
    )


def anthropic_semantic_query(request: Mapping[str, Any], namespace: str) -> SemanticQuery | None:
    """Semantic query of an Anthropic ``messages.create`` request (or None).

    The top-level ``system`` prompt is part of the exact-match context.
    """
    return _build_query(request, provider="anthropic", endpoint="messages", namespace=namespace)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticZones:
    """Similarity bounds: ``serve`` and above reuses, ``[refresh, serve)`` refreshes."""

    serve: float
    refresh: float

    def __post_init__(self) -> None:
        if not 0 < self.refresh <= self.serve <= 1:
            raise ValueError(
                "semantic thresholds must satisfy 0 < semantic_gray_threshold "
                f"<= semantic_threshold <= 1 (got {self.refresh} and {self.serve})"
            )

    def zone(self, score: float) -> str | None:
        """``"serve"``, ``"refresh"`` or None (a miss)."""
        if score >= self.serve:
            return "serve"
        if score >= self.refresh:
            return "refresh"
        return None


@dataclass(frozen=True)
class SemanticLookup:
    """Result of a lookup: the query embedding (for later storage) and any match.

    ``zone`` is ``"serve"`` (reuse ``payload``), ``"refresh"`` (call the API, then replace
    entry ``row_id``) or None (a miss).
    """

    embedding: EmbeddingResult
    payload: dict[str, Any] | None
    score: float | None
    stored_query: str | None = None
    zone: str | None = None
    row_id: int | None = None


class SemanticCache:
    """Embedding-similarity response cache.

    Args:
        store: Vector storage.
        embedder: Computes embeddings.
        threshold: Minimum cosine similarity to reuse an answer, in ``(0, 1]``.
        gray_threshold: Lower bound of the gray zone (default ``threshold - 0.05``).
        customer_thresholds: Per end customer ``(threshold, gray_threshold)`` overrides.
        ttl: Entry lifetime in seconds (None = never expires).
        verifier: Optional judge confirming each candidate before reuse.
    """

    def __init__(
        self,
        store: SQLiteVectorStore,
        embedder: Embedder | AsyncEmbedder,
        *,
        threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        gray_threshold: float | None = None,
        customer_thresholds: Mapping[str, tuple[float, float]] | None = None,
        ttl: float | None,
        verifier: Verifier | AsyncVerifier | None = None,
        floor: float | None = None,
    ) -> None:
        self.floor = check_floor(floor)
        if not 0 < threshold <= 1:
            raise ValueError("semantic_threshold must be in the interval (0, 1]")
        if gray_threshold is None:
            gray_threshold = max(round(threshold - DEFAULT_GRAY_MARGIN, 6), min(threshold, 0.01))
        self.default_zones = SemanticZones(serve=threshold, refresh=gray_threshold)
        self._customer_zones: dict[str, SemanticZones] = {}
        self._zones_lock = threading.Lock()
        for name, bounds in (customer_thresholds or {}).items():
            self.set_thresholds(name, *bounds)
        if not isinstance(embedder, (Embedder, AsyncEmbedder)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("semantic_embedder must provide embed() and/or aembed()")
        self.store = store
        self.embedder = embedder
        self.threshold = threshold  # default serve bound (kept for compatibility)
        self.ttl = ttl
        if verifier is not None and not isinstance(verifier, (Verifier, AsyncVerifier)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("semantic_verifier must provide verify() and/or averify()")
        self.verifier = verifier

    def embed(self, text: str) -> EmbeddingResult:
        """Compute an embedding synchronously."""
        if not isinstance(self.embedder, Embedder):
            raise TypeError("this embedder only supports asynchronous use (aembed)")
        return self.embedder.embed(text)

    async def aembed(self, text: str) -> EmbeddingResult:
        """Compute an embedding asynchronously (falls back to a worker thread)."""
        if isinstance(self.embedder, AsyncEmbedder):
            return await self.embedder.aembed(text)
        return await asyncio.to_thread(self.embedder.embed, text)

    # -- thresholds --------------------------------------------------------------

    def set_thresholds(self, customer: str, threshold: float, gray_threshold: float) -> None:
        """Set the serve and gray-zone bounds of one end customer (thread-safe)."""
        if not isinstance(customer, str) or not customer:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("the customer must be a non-empty string")
        zones = SemanticZones(serve=threshold, refresh=gray_threshold)
        with self._zones_lock:
            self._customer_zones[customer] = zones

    def reset_thresholds(self, customer: str) -> None:
        """Back to the default bounds for this customer."""
        with self._zones_lock:
            self._customer_zones.pop(customer, None)

    def zones_for(self, customer: str | None) -> SemanticZones:
        """Bounds in force for ``customer`` (the defaults when None or not set)."""
        return self.zones_with_source(customer)[0]

    def zones_with_source(self, customer: str | None) -> tuple[SemanticZones, str]:
        """Bounds in force for ``customer`` and their origin: ``"customer"`` or ``"default"``."""
        if customer is not None:
            with self._zones_lock:
                zones = self._customer_zones.get(customer)
            if zones is not None:
                return zones, "customer"
        return self.default_zones, "default"

    # -- lookups -----------------------------------------------------------------

    def search(
        self,
        query: SemanticQuery,
        embedding: EmbeddingResult,
        customer: str | None = None,
        max_age: float | None = None,
        zones: SemanticZones | None = None,
    ) -> SemanticLookup:
        """Find the closest stored answer for ``query`` and its zone (entries older than
        ``max_age`` seconds are ignored). ``zones`` overrides the customer's bounds."""
        if zones is None:
            zones = self.zones_for(customer)
        match = self.store.search(
            query.context_key, self.embedder.model, embedding.vector, zones.refresh, max_age
        )
        value: Any = None if match is None else match.value
        if match is None or not isinstance(value, dict):
            return SemanticLookup(embedding=embedding, payload=None, score=None)
        payload = cast("dict[str, Any]", value)
        return SemanticLookup(
            embedding=embedding,
            payload=payload,
            score=match.score,
            stored_query=match.query,
            zone=zones.zone(match.score),
            row_id=match.row_id,
        )

    def add(
        self,
        query: SemanticQuery,
        embedding: EmbeddingResult,
        payload: Any,
        replace_id: int | None = None,
        ttl: float | None = None,
        customer: str = "",
    ) -> None:
        """Store the answer to ``query`` (replacing entry ``replace_id`` if given)."""
        self.store.add(
            query.context_key,
            self.embedder.model,
            embedding.vector,
            payload,
            self.ttl if ttl is None else ttl,
            query=query.text,
            replace_id=replace_id,
            customer=customer,
        )
