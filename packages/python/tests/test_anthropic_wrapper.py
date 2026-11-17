"""The Anthropic wrapper must behave exactly like ``anthropic.Anthropic``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message
from typing_extensions import assert_type

import caelum
from caelum import Anthropic, AsyncAnthropic
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "How do I return an item?"}]
MODEL = "claude-sonnet-5-5"
# Fake server usage: 1,000 input + 500 output tokens ($2 / $10 per 1M).
SONNET_CALL_COST = (1000 * 2.00 + 500 * 10.00) / 1_000_000
_INTERNAL_CLIENT = "_caelum_client"


def make_client(server: FakeServer, **kwargs: Any) -> Anthropic:
    return Anthropic(api_key="sk-ant-test", http_client=server.http_client(), **kwargs)


def ask(c: Anthropic, **kwargs: Any) -> Message:
    params: dict[str, Any] = {"model": MODEL, "max_tokens": 1024, "messages": MESSAGES, **kwargs}
    return cast(Message, c.messages.create(**params))


def text(message: Message) -> str:
    block = message.content[0]
    assert block.type == "text"
    return block.text


def test_public_exports() -> None:
    from caelum.anthropic import AsyncCaelumAnthropic, CaelumAnthropic

    assert caelum.Anthropic is CaelumAnthropic
    assert caelum.AsyncAnthropic is AsyncCaelumAnthropic


def test_missing_anthropic_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "caelum.anthropic", raising=False)
    with pytest.raises(ImportError, match=r'pip install "caelum-ai\[anthropic\]"'):
        importlib.import_module("caelum.anthropic")


def test_message_is_identical_to_raw_sdk(server: FakeServer) -> None:
    raw = anthropic.Anthropic(api_key="sk-ant-test", http_client=server.http_client())
    expected = raw.messages.create(model=MODEL, max_tokens=1024, messages=MESSAGES)
    response = ask(make_client(server))
    assert type(response) is type(expected)
    assert text(response) == "answer #2"
    assert not caelum.cache_hit(response)


def test_arguments_are_forwarded_untouched(server: FakeServer) -> None:
    ask(make_client(server), system="You are a support agent.", stop_sequences=["END"])
    assert server.requests[-1]["body"] == {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": MESSAGES,
        "system": "You are a support agent.",
        "stop_sequences": ["END"],
    }


def test_streaming_with_stream_true(server: FakeServer) -> None:
    stream = make_client(server).messages.create(
        model=MODEL, max_tokens=1024, messages=MESSAGES, stream=True
    )
    deltas = [
        event.delta.text
        for event in stream
        if event.type == "content_block_delta" and event.delta.type == "text_delta"
    ]
    assert "".join(deltas) == "answer #1"


def test_stream_helper_is_forwarded(server: FakeServer) -> None:
    client = make_client(server)
    with client.messages.stream(model=MODEL, max_tokens=1024, messages=MESSAGES) as stream:
        final = stream.get_final_message()
    assert text(final) == "answer #1"


def test_other_methods_are_forwarded(server: FakeServer) -> None:
    client = make_client(server)
    assert hasattr(client.messages, "count_tokens")
    assert hasattr(client.messages, "batches")
    assert client.api_key == "sk-ant-test"
    assert client.models is getattr(client, _INTERNAL_CLIENT).models


def test_api_errors_keep_sdk_exception_types(server: FakeServer) -> None:
    server.failures = [400]
    with pytest.raises(anthropic.BadRequestError):
        ask(make_client(server))


def test_exact_cache(server: FakeServer, cache_path: Path) -> None:
    c = make_client(server, cache=True, cache_path=cache_path)
    ask(c, system="You are a support agent.")
    hit = ask(c, system="You are a support agent.")
    assert caelum.cache_hit(hit)
    assert isinstance(hit, Message)
    assert text(hit) == "answer #1"
    # A different system prompt must never reuse the answer.
    miss = ask(c, system="You are a pirate.")
    assert not caelum.cache_hit(miss)
    assert server.calls == 2


def test_openai_and_anthropic_never_share_cache_entries(
    server: FakeServer, cache_path: Path
) -> None:
    ask(make_client(server, cache=True, cache_path=cache_path))
    other = Anthropic(
        api_key="sk-ant-test",
        http_client=server.http_client(),
        base_url="https://other.example.com",
        cache=True,
        cache_path=cache_path,
    )
    assert not caelum.cache_hit(ask(other))


def test_metrics_and_cost(server: FakeServer, cache_path: Path) -> None:
    c = make_client(server, cache=True, cache_path=cache_path)
    ask(c)
    ask(c)
    s = c.metrics.summary()
    assert s["requests"] == 2
    assert s["cache_hits_exact"] == 1
    assert s["cost_usd"] == pytest.approx(SONNET_CALL_COST)
    assert s["cost_saved_usd"] == pytest.approx(SONNET_CALL_COST)


def test_copy_stays_wrapped(server: FakeServer) -> None:
    derived = make_client(server).with_options(max_retries=0)
    assert isinstance(derived, Anthropic)
    assert not caelum.cache_hit(ask(derived))


def test_static_types_match_the_sdk(server: FakeServer) -> None:
    client = make_client(server)
    response = client.messages.create(model=MODEL, max_tokens=1024, messages=MESSAGES)
    assert_type(response, Message)


async def test_async_messages_and_cache(server: FakeServer, cache_path: Path) -> None:
    async with AsyncAnthropic(
        api_key="sk-ant-test",
        http_client=server.async_http_client(),
        cache=True,
        cache_path=cache_path,
    ) as client:
        first = await client.messages.create(model=MODEL, max_tokens=1024, messages=MESSAGES)
        second = await client.messages.create(model=MODEL, max_tokens=1024, messages=MESSAGES)
    assert text(first) == "answer #1"
    assert caelum.cache_hit(second)
    assert client.metrics.summary()["requests"] == 2
