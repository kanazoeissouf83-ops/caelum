"""Fallback: when a provider is down, users do not notice — and errors stay honest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anthropic
import openai
import pytest
from openai.types.chat import ChatCompletion

import caelum
from caelum import AsyncOpenAI, FallbackChain, OpenAI, RetryPolicy
from caelum.fallback import FallbackTier, should_fallback
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "Where is my order?"}]


def client(server: FakeServer, fallback: list[Any], **kwargs: Any) -> OpenAI:
    raw_anthropic = anthropic.Anthropic(
        api_key="sk-ant-test", http_client=server.http_client(), max_retries=0
    )
    options: dict[str, Any] = {
        "fallback": fallback,
        "provider_clients": {"anthropic": raw_anthropic},
        "max_retries": 0,  # no SDK-level retries in tests
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: OpenAI, **kwargs: Any) -> Any:
    params: dict[str, Any] = {"model": "gpt-4o", "messages": MESSAGES, **kwargs}
    result: Any = cast(Any, c.chat.completions.create(**params))
    return result


def test_outage_switches_to_anthropic_transparently(server: FakeServer) -> None:
    server.failures = [503]
    c = client(server, ["anthropic:claude-sonnet-5-5"])
    response = ask(c)
    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "answer #1"
    assert caelum.routed_to(response) == "claude-sonnet-5-5"
    assert server.requests[-1]["path"].endswith("/messages")
    s = c.metrics.summary()
    assert s["fallbacks"] == 1
    assert s["failed_requests"] == 0
    assert s["cost_usd"] == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)


def test_rate_limit_triggers_fallback(server: FakeServer) -> None:
    server.failures = [429]
    response = ask(client(server, [{"provider": "anthropic", "model": "claude-haiku-5-5"}]))
    assert caelum.routed_to(response) == "claude-haiku-5-5"


def test_client_errors_never_trigger_fallback(server: FakeServer) -> None:
    server.failures = [400]
    with pytest.raises(openai.BadRequestError):
        ask(client(server, ["anthropic:claude-sonnet-5-5"]))
    assert len(server.requests) == 1


def test_original_error_is_raised_when_every_tier_fails(server: FakeServer) -> None:
    server.failures = [503, 503, 503]
    c = client(server, ["anthropic:claude-sonnet-5-5", "openai:gpt-4o-mini"])
    with pytest.raises(openai.InternalServerError):
        ask(c)
    assert len(server.requests) == 3
    assert c.metrics.summary()["failed_requests"] == 1


def test_same_provider_tier_supports_streaming(server: FakeServer) -> None:
    server.failures = [503]
    c = client(server, ["anthropic:claude-sonnet-5-5", "openai:gpt-4o-mini"])
    stream = ask(c, stream=True)  # the anthropic tier is skipped for streams
    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "answer #1"
    assert server.requests[-1]["body"]["model"] == "gpt-4o-mini"


def test_fallback_answers_are_not_cached(server: FakeServer, cache_path: Path) -> None:
    server.failures = [503]
    c = client(server, ["anthropic:claude-sonnet-5-5"], cache=True, cache_path=cache_path)
    ask(c)
    again = ask(c)  # primary is back: must be asked again
    assert not caelum.cache_hit(again)
    assert server.requests[-1]["path"].endswith("/chat/completions")
    assert caelum.cache_hit(ask(c))  # the primary answer is cached normally


def test_retries_happen_before_fallback(server: FakeServer) -> None:
    server.failures = [503, 503, 503]
    c = client(
        server,
        ["anthropic:claude-sonnet-5-5"],
        retry=RetryPolicy(max_retries=2, initial_delay=0, jitter=0),
    )
    response = ask(c)
    assert caelum.routed_to(response) == "claude-sonnet-5-5"
    assert c.metrics.summary()["retries"] == 2


def test_untranslatable_requests_skip_cross_provider_tiers(server: FakeServer) -> None:
    server.failures = [503]
    with pytest.raises(openai.InternalServerError):
        ask(client(server, ["anthropic:claude-sonnet-5-5"]), n=2)


async def test_async_fallback(server: FakeServer) -> None:
    server.failures = [502]
    raw_anthropic = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", http_client=server.async_http_client(), max_retries=0
    )
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        max_retries=0,
        fallback=["anthropic:claude-haiku-5-5"],
        provider_clients={"anthropic": raw_anthropic},
    )
    response = await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert caelum.routed_to(response) == "claude-haiku-5-5"


async def test_async_all_tiers_fail(server: FakeServer) -> None:
    server.failures = [502, 502]
    raw_openai_tier = "openai:gpt-4o-mini"
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        max_retries=0,
        fallback=[raw_openai_tier],
    )
    with pytest.raises(openai.InternalServerError):
        await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)


# -- configuration ------------------------------------------------------------------


INVALID_CHAINS: list[Any] = [
    ["anthropic"],  # no model: CAELUM never picks one for you
    ["google:gemini-pro"],
    [{"provider": "anthropic"}],
    [{"provider": "anthropic", "model": "m", "weight": 2}],
    [],
]


@pytest.mark.parametrize("spec", INVALID_CHAINS)
def test_invalid_chains_are_rejected(spec: Any) -> None:
    with pytest.raises(ValueError):
        FallbackChain.parse(spec)


def test_chain_parsing() -> None:
    chain = FallbackChain.parse(
        ["anthropic:claude-sonnet-5-5", {"provider": "openai", "model": "m"}]
    )
    assert chain.tiers == (
        FallbackTier("anthropic", "claude-sonnet-5-5"),
        FallbackTier("openai", "m"),
    )
    assert FallbackChain.parse(chain) is chain
    assert str(chain.tiers[0]) == "anthropic:claude-sonnet-5-5"
    with pytest.raises(TypeError):
        FallbackChain.parse("anthropic:claude-sonnet-5-5")
    with pytest.raises(TypeError):
        FallbackChain.parse([42])


def test_should_fallback() -> None:
    class StatusError(Exception):
        def __init__(self, code: int) -> None:
            self.status_code = code

    assert should_fallback(StatusError(503))
    assert should_fallback(StatusError(429))
    assert not should_fallback(StatusError(400))
    assert not should_fallback(ValueError())
