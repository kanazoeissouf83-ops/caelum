"""Intelligent retries for transient provider errors.

Opt-in on any CAELUM client::

    client = OpenAI(retry=True)                                   # default policy
    client = OpenAI(retry=RetryPolicy(max_retries=5, max_delay=60))

Behaviour:

* retries rate limits (429), server errors (500, 502, 503, 504), Anthropic overload
  (529), timeouts and network errors;
* never retries client errors (400, 401, 403, 404, 422...);
* exponential backoff (``initial_delay * 2**attempt``) plus random jitter, capped by
  ``max_delay``; a ``Retry-After`` / ``retry-after-ms`` header from the provider wins;
* when every attempt fails, the **original SDK exception** is re-raised (so existing
  ``except openai.RateLimitError`` code keeps working), with the attempt history attached.

WHY a separate ``retry`` option instead of the planned ``max_retries``: the provider SDKs
already accept ``max_retries`` (their own built-in retries, 2 by default). Shadowing it
would break the drop-in guarantee. When CAELUM retries are enabled, the SDK's built-in
retries are turned off (unless ``max_retries`` is passed explicitly) so attempts never
multiply.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from caelum.utils import logger

__all__ = [
    "DEFAULT_RETRYABLE_STATUS_CODES",
    "RetryPolicy",
    "async_with_retry",
    "is_retryable",
    "retry_after_seconds",
    "with_retry",
]

T = TypeVar("T")

DEFAULT_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
# Exception classes (matched by name, for every provider SDK) meaning "network problem".
# OpenAI/Anthropic connection errors, plus the raw httpx transport errors that other SDKs
# (Google Gen AI) let through.
_CONNECTION_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    """How CAELUM retries transient failures.

    Attributes:
        max_retries: Retries after the first attempt (total attempts = max_retries + 1).
        initial_delay: Delay before the first retry, in seconds.
        max_delay: Upper bound for any single delay, in seconds.
        jitter: Random extra delay in ``[0, jitter]`` seconds (avoids retry storms).
        retryable_status_codes: HTTP status codes worth retrying.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 1.0
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_STATUS_CODES
    )

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.initial_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("delays and jitter must be >= 0")

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before retry number ``attempt`` (0-based), in seconds."""
        if retry_after is not None:
            return min(self.max_delay, max(0.0, retry_after))
        exponential = self.initial_delay * (2**attempt)
        return min(self.max_delay, exponential + random.uniform(0, self.jitter))


def coerce_policy(retry: bool | RetryPolicy | None) -> RetryPolicy | None:
    """Turn the ``retry=`` client option into a policy (None = disabled)."""
    if retry is None or retry is False:
        return None
    if retry is True:
        return RetryPolicy()
    if isinstance(retry, RetryPolicy):  # pyright: ignore[reportUnnecessaryIsInstance]
        return retry
    raise TypeError("retry must be True, False, None or a caelum.RetryPolicy")


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    # Google Gen AI errors expose the HTTP status as ``code``.
    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599:
        return code
    return None


def is_retryable(exc: BaseException, policy: RetryPolicy) -> bool:
    """Return True if ``exc`` is a transient error worth retrying."""
    status = _status_code(exc)
    if status is not None:
        return status in policy.retryable_status_codes
    if any(cls.__name__ in _CONNECTION_ERROR_NAMES for cls in type(exc).__mro__):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read the provider's ``retry-after-ms`` / ``retry-after`` header, if any."""
    response = getattr(exc, "response", None)
    headers: Any = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        milliseconds = headers.get("retry-after-ms")
        if milliseconds is not None:
            return float(milliseconds) / 1000
        value = headers.get("retry-after")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            moment = email.utils.parsedate_to_datetime(str(value))
            return moment.timestamp() - time.time()
    except (TypeError, ValueError, AttributeError):
        return None


def _describe(exc: BaseException) -> str:
    status = _status_code(exc)
    return f"{type(exc).__name__}({status})" if status is not None else type(exc).__name__


def _give_up(exc: BaseException, history: list[str]) -> None:
    """Attach the attempt history to the exception that is about to be re-raised."""
    try:
        object.__setattr__(exc, "_caelum_retry_history", list(history))
    except (AttributeError, TypeError):  # pragma: no cover
        pass
    if sys.version_info >= (3, 11):
        exc.add_note(f"caelum: gave up after {len(history)} attempts: {', '.join(history)}")


def _log_retry(exc: BaseException, attempt: int, delay: float) -> None:
    logger.warning(
        "caelum: transient error %s, retry %d in %.2fs",
        _describe(exc),
        attempt + 1,
        delay,
        extra={
            "caelum_event": "retry",
            "caelum_attempt": attempt + 1,
            "caelum_delay_s": delay,
            "caelum_status": _status_code(exc),
            "caelum_error": type(exc).__name__,
        },
    )


def with_retry(
    func: Callable[[], T],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Call ``func`` and retry it on transient errors according to ``policy``.

    Args:
        func: Zero-argument callable performing one attempt.
        policy: Retry policy.
        on_retry: Called before each retry (used for metrics).
        sleep: Sleep function (injectable for tests). Defaults to ``time.sleep``.

    Raises:
        The last exception raised by ``func`` once retries are exhausted, or immediately
        for non-retryable errors.
    """
    pause = sleep if sleep is not None else time.sleep
    history: list[str] = []
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            history.append(_describe(exc))
            if not is_retryable(exc, policy) or attempt >= policy.max_retries:
                if attempt > 0:
                    _give_up(exc, history)
                raise
            delay = policy.backoff(attempt, retry_after_seconds(exc))
            _log_retry(exc, attempt, delay)
            if on_retry is not None:
                on_retry()
            pause(delay)
            attempt += 1


async def async_with_retry(
    func: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Asynchronous counterpart of :func:`with_retry`."""
    pause = sleep if sleep is not None else asyncio.sleep
    history: list[str] = []
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as exc:
            history.append(_describe(exc))
            if not is_retryable(exc, policy) or attempt >= policy.max_retries:
                if attempt > 0:
                    _give_up(exc, history)
                raise
            delay = policy.backoff(attempt, retry_after_seconds(exc))
            _log_retry(exc, attempt, delay)
            if on_retry is not None:
                on_retry()
            await pause(delay)
            attempt += 1
