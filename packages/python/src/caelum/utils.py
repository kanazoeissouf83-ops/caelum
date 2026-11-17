"""Shared internal helpers: logging, canonical hashing, response annotation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

logger = logging.getLogger("caelum")

# Names of the "argument not provided" sentinel classes used by the OpenAI and
# Anthropic SDKs (``NOT_GIVEN`` / ``omit``) and the Mistral SDK (``UNSET``).
_SENTINEL_CLASS_NAMES = frozenset({"NotGiven", "Omit", "Unset"})


class UnhashableRequestError(TypeError):
    """Raised when a request contains a value that cannot be canonically serialized."""


# Extras whose package has another name than the extra itself.
_EXTRA_PACKAGES = {
    "gemini": "google-genai",
    "mistral": "mistralai",
    "yaml": "pyyaml",
    "fast": "numpy",
}


def missing_extra_message(extra: str, module: str | None = None) -> str:
    """Build the error message shown when an optional extra is not installed.

    Args:
        extra: Name of the extra (and of the missing package), e.g. ``"openai"``.
        module: CAELUM module requiring it. Defaults to ``caelum.<extra>``.

    Returns:
        A human-readable message including the exact install command.
    """
    where = module or f"caelum.{extra}"
    package = _EXTRA_PACKAGES.get(extra, extra)
    return (
        f"{where} requires the '{package}' package, which is not installed. "
        f'Install it with: pip install "caelum-ai[{extra}]"'
    )


def is_sentinel(value: object) -> bool:
    """Return True if ``value`` is an SDK "not given" sentinel (NOT_GIVEN / omit)."""
    return type(value).__name__ in _SENTINEL_CLASS_NAMES


def to_canonical(value: Any) -> Any:
    """Convert a request value into plain JSON-compatible data.

    Sentinels are dropped from mappings, pydantic models are dumped, tuples become
    lists. Anything else that is not JSON-compatible raises.

    Raises:
        UnhashableRequestError: If a value cannot be represented deterministically.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, type):
        # WHY: a class (e.g. a pydantic response schema) has an unbound ``model_dump``;
        # it has no stable JSON form, so such requests are simply not cached.
        raise UnhashableRequestError(f"class objects are not hashable: {value.__name__}")
    if isinstance(value, float):
        # WHY: NaN/inf have no canonical JSON form; refuse rather than guess.
        if value != value or value in (float("inf"), float("-inf")):
            raise UnhashableRequestError(f"non-finite float: {value!r}")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        items = cast("Mapping[object, Any]", value).items()
        for key, item in items:
            if not isinstance(key, str):
                raise UnhashableRequestError(f"non-string key: {key!r}")
            if is_sentinel(item):
                continue
            out[key] = to_canonical(item)
        return out
    if isinstance(value, (list, tuple)):
        return [to_canonical(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", exclude_unset=True)
        except Exception as exc:
            # WHY: e.g. a Gemini config holding Python functions as tools. Such requests
            # are not cached rather than breaking the call.
            raise UnhashableRequestError(f"unserializable {type(value).__name__}") from exc
        return to_canonical(dumped)
    raise UnhashableRequestError(f"unsupported value of type {type(value).__name__}")


def stable_hash(data: Any) -> str:
    """Return a deterministic SHA-256 hex digest of JSON-compatible ``data``."""
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def credential_scope(*secrets: object) -> str:
    """Return a namespace suffix that isolates cached answers per credential.

    Two clients using different API keys (e.g. one per customer in a multi-tenant
    app) must never be served each other's answers. Only a truncated SHA-256 of the
    secret is kept: the key itself is never written to the cache file.
    """
    parts = [str(secret) for secret in secrets if isinstance(secret, str) and secret]
    if not parts:
        return ""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"|cred:{digest}"


def memoized_namespace(
    inputs: Callable[[], tuple[Any, ...]], build: Callable[[tuple[Any, ...]], str]
) -> Callable[[], str]:
    """Namespace function recomputed only when its (cheap) inputs change.

    WHY: the namespace follows the live client (``client.api_key = ...``) but hashing on
    every request would add latency; the inputs are plain attribute reads.
    """
    last: list[tuple[tuple[Any, ...], str] | None] = [None]

    def namespace() -> str:
        key = inputs()
        cached = last[0]  # one read: the (key, value) pair is replaced atomically
        if cached is not None and cached[0] == key:
            return cached[1]
        value = build(key)
        last[0] = (key, value)
        return value

    return namespace


def extract_usage(response: object) -> tuple[int, int] | None:
    """Return ``(input_tokens, output_tokens)`` from a provider response.

    Supports OpenAI (``prompt_tokens`` / ``completion_tokens``), Anthropic
    (``input_tokens`` / ``output_tokens``) and Gemini (``usage_metadata``, where thinking
    tokens are billed as output). Returns None if absent.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return _gemini_usage(response)
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)
    if not isinstance(input_tokens, int):
        return None
    return input_tokens, output_tokens if isinstance(output_tokens, int) else 0


def _gemini_usage(response: object) -> tuple[int, int] | None:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None
    prompt = getattr(metadata, "prompt_token_count", None)
    if not isinstance(prompt, int):
        return None
    output = 0
    for name in ("candidates_token_count", "thoughts_token_count"):
        count = getattr(metadata, name, None)
        if isinstance(count, int):
            output += count
    return prompt, output


def served_model(response: object) -> str | None:
    """Model name reported by a provider response (``model`` or Gemini ``model_version``)."""
    for name in ("model", "model_version"):
        value = getattr(response, name, None)
        if isinstance(value, str) and value:
            return value.removeprefix("models/")
    return None


def format_usd(amount: float) -> str:
    """Dollar amount for humans: cents above one cent, more digits below.

    A request to a small model costs a fraction of a cent: ``$ 0.00`` would hide it.
    """
    if amount == 0 or abs(amount) >= 1:
        return f"$ {amount:.2f}"
    if abs(amount) >= 0.01:
        return f"$ {amount:.4f}"
    return f"$ {amount:.6f}"


def cache_hit(response: object) -> bool:
    """Return True if ``response`` was served from a CAELUM cache.

    Example::

        response = client.chat.completions.create(...)
        if caelum.cache_hit(response):
            ...
    """
    return bool(getattr(response, "_caelum_cache_hit", False))


def cache_type(response: object) -> str | None:
    """Return ``"exact"``, ``"semantic"`` or ``None`` (not served from cache)."""
    value = getattr(response, "_caelum_cache_type", None)
    return value if isinstance(value, str) else None


def annotate_response(response: object, *, cache_hit: bool, cache_type: str | None) -> None:
    """Attach CAELUM debug attributes to a provider response, best effort.

    Sets ``_caelum_cache_hit`` (bool) and ``_caelum_cache_type``
    (``"exact"``, ``"semantic"`` or ``None``). Never raises.
    """
    try:
        object.__setattr__(response, "_caelum_cache_hit", cache_hit)
        object.__setattr__(response, "_caelum_cache_type", cache_type)
    except (AttributeError, TypeError):  # pragma: no cover - objects without __dict__
        logger.debug("caelum: could not annotate response of type %s", type(response))


def annotate_route(
    response: object,
    *,
    routed_from: str | None,
    routed_to: str,
    provider: str,
    route: str | None,
) -> None:
    """Attach routing/fallback debug attributes to a response, best effort.

    Sets ``_caelum_routed_from`` (requested model), ``_caelum_routed_to`` (model that
    answered), ``_caelum_provider`` and ``_caelum_route`` (rule name, None for fallback).
    """
    try:
        object.__setattr__(response, "_caelum_routed_from", routed_from)
        object.__setattr__(response, "_caelum_routed_to", routed_to)
        object.__setattr__(response, "_caelum_provider", provider)
        object.__setattr__(response, "_caelum_route", route)
    except (AttributeError, TypeError):  # pragma: no cover - objects without __dict__
        logger.debug("caelum: could not annotate response of type %s", type(response))


def routed_to(response: object) -> str | None:
    """Model that actually answered, if CAELUM routed or fell back (else None)."""
    value = getattr(response, "_caelum_routed_to", None)
    return value if isinstance(value, str) else None
