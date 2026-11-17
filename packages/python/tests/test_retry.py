"""Retries: transient errors are retried, client errors never are, drop-in preserved."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import anthropic
import openai
import pytest

from caelum import Anthropic, AsyncOpenAI, OpenAI, RetryPolicy
from caelum.retry import (
    async_with_retry,
    coerce_policy,
    is_retryable,
    retry_after_seconds,
    with_retry,
)
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "Hi"}]
FAST = RetryPolicy(max_retries=3, initial_delay=0, jitter=0)
HISTORY_ATTR = "_caelum_retry_history"


def client(server: FakeServer, **kwargs: Any) -> OpenAI:
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **kwargs)


def ask(c: OpenAI) -> Any:
    return c.chat.completions.create(model="gpt-4o", messages=MESSAGES)


class StatusError(Exception):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


class APIConnectionError(Exception):
    """Same class name as the SDKs' network error."""


class APITimeoutError(APIConnectionError):
    pass


# -- end-to-end through the real SDKs --------------------------------------------


def test_rate_limit_is_retried_until_success(server: FakeServer) -> None:
    server.failures = [429, 429]
    c = client(server, retry=FAST)
    response = ask(c)
    assert response.choices[0].message.content == "answer #1"
    assert len(server.requests) == 3
    s = c.metrics.summary()
    assert s["retries"] == 2
    assert s["requests"] == 1
    assert s["failed_requests"] == 0


def test_exhausted_retries_reraise_the_original_sdk_exception(server: FakeServer) -> None:
    server.failures = [503, 503, 503, 503]
    c = client(server, retry=FAST)
    with pytest.raises(openai.InternalServerError) as caught:
        ask(c)
    assert len(server.requests) == 4  # 1 attempt + 3 retries
    history: list[str] = getattr(caught.value, HISTORY_ATTR)
    assert history == ["InternalServerError(503)"] * 4
    if sys.version_info >= (3, 11):
        assert "gave up after 4 attempts" in " ".join(caught.value.__notes__)
    assert c.metrics.summary()["failed_requests"] == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_never_retried(server: FakeServer, status: int) -> None:
    server.failures = [status]
    with pytest.raises(openai.APIStatusError):
        ask(client(server, retry=FAST))
    assert len(server.requests) == 1


def test_retry_after_header_zero_is_honoured(server: FakeServer) -> None:
    server.failures = [429]
    server.failure_headers = {"retry-after": "0"}
    slow_policy = RetryPolicy(max_retries=1, initial_delay=60, jitter=0)
    ask(client(server, retry=slow_policy))  # would take 60 s without the header
    assert len(server.requests) == 2


def test_retries_are_disabled_by_default(server: FakeServer) -> None:
    c = client(server)
    assert c.max_retries == openai.DEFAULT_MAX_RETRIES  # SDK behaviour untouched
    server.failures = [400]
    with pytest.raises(openai.BadRequestError):
        ask(c)


def test_enabling_retries_turns_off_sdk_retries(server: FakeServer) -> None:
    assert client(server, retry=True).max_retries == 0
    assert client(server, retry=True, max_retries=5).max_retries == 5  # explicit wins


def test_anthropic_overload_529_is_retried(server: FakeServer) -> None:
    server.failures = [529]
    c = Anthropic(api_key="sk-ant-test", http_client=server.http_client(), retry=FAST)
    message = c.messages.create(model="claude-haiku-5-5", max_tokens=64, messages=MESSAGES)
    assert message.content[0].type == "text"
    assert len(server.requests) == 2
    assert isinstance(anthropic.Anthropic, type)


async def test_async_retry(server: FakeServer) -> None:
    server.failures = [502]
    c = AsyncOpenAI(api_key="sk-test", http_client=server.async_http_client(), retry=FAST)
    response = await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert response.choices[0].message.content == "answer #1"
    assert c.metrics.summary()["retries"] == 1


# -- unit tests ------------------------------------------------------------------


def test_is_retryable() -> None:
    policy = RetryPolicy()
    for status in (429, 500, 502, 503, 504, 529):
        assert is_retryable(StatusError(status), policy)
    for status in (400, 401, 403, 404, 422):
        assert not is_retryable(StatusError(status), policy)
    assert is_retryable(APIConnectionError(), policy)
    assert is_retryable(APITimeoutError(), policy)
    assert is_retryable(ConnectionResetError(), policy)
    assert is_retryable(TimeoutError(), policy)
    assert not is_retryable(ValueError(), policy)
    assert is_retryable(StatusError(418), RetryPolicy(retryable_status_codes=frozenset({418})))


def test_retry_after_parsing() -> None:
    assert retry_after_seconds(StatusError(429, {"retry-after": "3"})) == 3
    assert retry_after_seconds(StatusError(429, {"retry-after-ms": "1500"})) == 1.5
    http_date = retry_after_seconds(
        StatusError(429, {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    )
    assert http_date is not None and http_date < 0  # a date in the past
    assert retry_after_seconds(StatusError(429, {"retry-after": "soon"})) is None
    assert retry_after_seconds(StatusError(429)) is None
    assert retry_after_seconds(ValueError()) is None


def test_backoff_is_exponential_capped_and_jittered() -> None:
    policy = RetryPolicy(initial_delay=1, max_delay=5, jitter=0)
    assert [policy.backoff(i) for i in range(5)] == [1, 2, 4, 5, 5]
    assert policy.backoff(0, retry_after=3) == 3
    assert policy.backoff(0, retry_after=100) == 5  # capped by max_delay
    jittered = RetryPolicy(initial_delay=1, max_delay=100, jitter=1)
    for _ in range(50):
        assert 1 <= jittered.backoff(0) <= 2


def test_with_retry_uses_retry_after_and_counts_retries() -> None:
    delays: list[float] = []
    retries: list[int] = []
    errors = [StatusError(429, {"retry-after": "2"}), StatusError(500)]

    def attempt() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    policy = RetryPolicy(max_retries=3, initial_delay=10, max_delay=30, jitter=0)
    result = with_retry(attempt, policy, on_retry=lambda: retries.append(1), sleep=delays.append)
    assert result == "ok"
    assert delays == [2, 20]  # header first, then exponential backoff (10 * 2**1)
    assert len(retries) == 2


def test_with_retry_raises_non_retryable_immediately() -> None:
    calls: list[int] = []

    def attempt() -> None:
        calls.append(1)
        raise StatusError(400)

    with pytest.raises(StatusError):
        with_retry(attempt, FAST, sleep=lambda _: None)
    assert len(calls) == 1


async def test_async_with_retry() -> None:
    errors = [APIConnectionError()]
    delays: list[float] = []

    async def attempt() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    assert await async_with_retry(attempt, FAST, sleep=fake_sleep) == "ok"
    assert delays == [0]


def test_policy_validation_and_coercion() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(initial_delay=-1)
    assert coerce_policy(None) is None
    assert coerce_policy(False) is None
    assert coerce_policy(True) == RetryPolicy()
    assert coerce_policy(FAST) is FAST
    with pytest.raises(TypeError):
        coerce_policy(3)  # pyright: ignore[reportArgumentType]
