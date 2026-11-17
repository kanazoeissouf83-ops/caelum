"""The Mistral wrapper must behave exactly like ``mistralai.client.Mistral``.

The real Mistral SDK is plugged into an in-memory fake server through
``httpx.MockTransport``.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from mistralai.client import Mistral as RawMistral
from mistralai.client import errors as mistral_errors
from mistralai.client.models import ChatCompletionResponse
from typing_extensions import assert_type

import caelum
from caelum import Mistral, RetryPolicy

MODEL = "mistral-small-latest"
MESSAGES: list[Any] = [{"role": "user", "content": "How do I return an item?"}]
# Fake usage: 1,000 prompt + 500 completion tokens ($0.15 / $0.60 per 1M).
CALL_COST = (1000 * 0.15 + 500 * 0.60) / 1_000_000
FAST = RetryPolicy(max_retries=3, initial_delay=0, jitter=0)


@dataclass
class FakeMistral:
    """In-memory fake of the Mistral ``/v1/chat/completions`` endpoint."""

    calls: int = 0
    failures: list[int] = field(default_factory=lambda: [])
    bodies: list[dict[str, Any]] = field(default_factory=lambda: [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.bodies.append(body)
        if self.failures:
            status = self.failures.pop(0)
            return httpx.Response(status, json={"message": f"simulated {status}"})
        self.calls += 1
        return httpx.Response(
            200,
            json={
                "id": f"cmpl-{self.calls}",
                "object": "chat.completion",
                "model": body.get("model", MODEL),
                "created": 1_700_000_000,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"answer #{self.calls}"},
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            },
        )

    def clients(self) -> dict[str, Any]:
        async def ahandler(request: httpx.Request) -> httpx.Response:
            return self.handler(request)

        return {
            "client": httpx.Client(transport=httpx.MockTransport(self.handler)),
            "async_client": httpx.AsyncClient(transport=httpx.MockTransport(ahandler)),
        }


@pytest.fixture
def fake() -> FakeMistral:
    return FakeMistral()


def make_client(fake: FakeMistral, **kwargs: Any) -> Mistral:
    return Mistral(api_key="test-key", **fake.clients(), **kwargs)


def ask(client: Mistral, **kwargs: Any) -> ChatCompletionResponse:
    params: dict[str, Any] = {"model": MODEL, "messages": MESSAGES, **kwargs}
    return client.chat.complete(**params)


def text(response: ChatCompletionResponse) -> str:
    message = response.choices[0].message
    assert message is not None
    content = message.content
    assert isinstance(content, str)
    return content


# -- drop-in behaviour -----------------------------------------------------------


def test_public_export() -> None:
    from caelum.mistral import CaelumMistral

    assert caelum.Mistral is CaelumMistral


def test_missing_mistral_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mistralai.client", None)
    monkeypatch.delitem(sys.modules, "caelum.mistral", raising=False)
    with pytest.raises(ImportError, match=r'pip install "caelum-ai\[mistral\]"'):
        importlib.import_module("caelum.mistral")


def test_response_is_identical_to_raw_sdk(fake: FakeMistral) -> None:
    raw = RawMistral(api_key="test-key", **fake.clients())
    expected = raw.chat.complete(model=MODEL, messages=MESSAGES)
    response = make_client(fake).chat.complete(model=MODEL, messages=MESSAGES)
    assert_type(response, ChatCompletionResponse)
    assert type(response) is type(expected)
    assert text(response) == "answer #2"
    assert not caelum.cache_hit(response)


def test_other_resources_are_forwarded(fake: FakeMistral) -> None:
    client = make_client(fake)
    assert "stream" in dir(client.chat)
    assert client.chat.stream.__func__ is type(client._caelum_client.chat).stream  # type: ignore[attr-defined]  # noqa: SLF001
    assert client.embeddings is client._caelum_client.embeddings  # type: ignore[attr-defined]  # noqa: SLF001


def test_context_managers(fake: FakeMistral) -> None:
    with make_client(fake) as client:
        assert isinstance(client, Mistral)
        assert text(ask(client)) == "answer #1"


async def test_async_context_manager(fake: FakeMistral) -> None:
    async with make_client(fake) as client:
        assert isinstance(client, Mistral)


def test_positional_arguments_reach_the_sdk_unchanged(fake: FakeMistral) -> None:
    with pytest.raises(TypeError):
        make_client(fake).chat.complete(MODEL, MESSAGES)  # type: ignore[misc]


# -- cache -------------------------------------------------------------------------


def test_exact_cache_hit_and_metrics(fake: FakeMistral, cache_path: Path) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    first = ask(client)
    second = ask(client)
    assert fake.calls == 1
    assert text(second) == text(first) == "answer #1"
    assert type(second) is ChatCompletionResponse
    assert caelum.cache_hit(second) and caelum.cache_type(second) == "exact"
    summary = client.metrics.summary()
    assert summary["cache_hits"] == 1
    assert summary["cost_usd"] == pytest.approx(CALL_COST)
    assert summary["cost_saved_usd"] == pytest.approx(CALL_COST)


def test_parameters_change_the_key_but_transport_options_do_not(
    fake: FakeMistral, cache_path: Path
) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    ask(client)
    ask(client, temperature=0.1)
    hit = ask(client, timeout_ms=5000, http_headers={"x-trace": "1"})
    assert fake.calls == 2
    assert caelum.cache_hit(hit)


def test_cache_disabled_by_default(fake: FakeMistral) -> None:
    client = make_client(fake)
    ask(client)
    ask(client)
    assert fake.calls == 2


async def test_async_shares_cache_and_metrics(fake: FakeMistral, cache_path: Path) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    first = await client.chat.complete_async(model=MODEL, messages=MESSAGES)
    assert_type(first, ChatCompletionResponse)
    again = ask(client)
    assert fake.calls == 1
    assert caelum.cache_hit(again)


# -- retries -------------------------------------------------------------------------


def test_retry_on_rate_limit(fake: FakeMistral) -> None:
    fake.failures = [429, 503]
    client = make_client(fake, retry=FAST)
    assert text(ask(client)) == "answer #1"
    assert client.metrics.summary()["retries"] == 2


async def test_async_retry(fake: FakeMistral) -> None:
    fake.failures = [500]
    client = make_client(fake, retry=FAST)
    response = await client.chat.complete_async(model=MODEL, messages=MESSAGES)
    assert text(response) == "answer #1"


def test_client_errors_are_not_retried_and_keep_sdk_type(fake: FakeMistral) -> None:
    fake.failures = [401]
    client = make_client(fake, retry=FAST)
    with pytest.raises(mistral_errors.SDKError) as info:
        ask(client)
    assert info.value.status_code == 401
    assert fake.calls == 0
    assert client.metrics.summary()["failed_requests"] == 1
