"""The Gemini wrapper must behave exactly like ``google.genai.Client``.

The real Google Gen AI SDK is plugged into an in-memory fake server through
``httpx.MockTransport`` (this SDK uses httpx, not httpx2).
"""

# WHY: the SDK's own ``generate_content`` signature mentions PIL images, which pyright
# cannot resolve when Pillow is not installed. Same result with the raw SDK.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from google import genai as google_genai
from google.genai import errors as google_errors
from google.genai import types as google_types
from pydantic import BaseModel
from typing_extensions import assert_type

import caelum
from caelum import RetryPolicy, genai
from caelum.retry import is_retryable
from caelum.utils import extract_usage

MODEL = "gemini-2.5-flash"
# Fake usage: 1,000 prompt + 400 candidate + 100 thinking tokens ($0.30 / $2.50 per 1M).
CALL_COST = (1000 * 0.30 + 500 * 2.50) / 1_000_000
FAST = RetryPolicy(max_retries=3, initial_delay=0, jitter=0)


@dataclass
class FakeGemini:
    """In-memory fake of the Gemini ``generateContent`` endpoint."""

    calls: int = 0
    failures: list[int] = field(default_factory=lambda: [])
    bodies: list[dict[str, Any]] = field(default_factory=lambda: [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content) if request.content else {})
        if self.failures:
            status = self.failures.pop(0)
            return httpx.Response(
                status, json={"error": {"code": status, "message": "simulated", "status": "X"}}
            )
        self.calls += 1
        model = request.url.path.split("/models/")[-1].split(":")[0]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": f"answer #{self.calls}"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 400,
                    "thoughtsTokenCount": 100,
                    "totalTokenCount": 1500,
                },
                "modelVersion": model,
                "responseId": f"r{self.calls}",
            },
        )

    def http_options(self) -> google_types.HttpOptions:
        transport = httpx.MockTransport(self.handler)

        async def ahandler(request: httpx.Request) -> httpx.Response:
            return self.handler(request)

        return google_types.HttpOptions(
            httpx_client=httpx.Client(transport=transport),
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(ahandler)),
        )


@pytest.fixture
def fake() -> FakeGemini:
    return FakeGemini()


def make_client(fake: FakeGemini, **kwargs: Any) -> genai.Client:
    return genai.Client(api_key="test-key", http_options=fake.http_options(), **kwargs)


def ask(client: genai.Client, contents: Any = "How do I return an item?", **kwargs: Any) -> Any:
    return client.models.generate_content(model=MODEL, contents=contents, **kwargs)


# -- drop-in behaviour -----------------------------------------------------------


def test_module_mirrors_google_genai() -> None:
    assert caelum.genai is genai
    assert genai.types is google_types
    assert genai.errors is google_errors
    assert genai.Client is not google_genai.Client
    assert "genai" in dir(caelum)


def test_missing_gemini_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.delitem(sys.modules, "caelum.genai", raising=False)
    with pytest.raises(ImportError, match=r'pip install "caelum-ai\[gemini\]"'):
        importlib.import_module("caelum.genai")


def test_response_is_identical_to_raw_sdk(fake: FakeGemini) -> None:
    raw = google_genai.Client(api_key="test-key", http_options=fake.http_options())
    expected = raw.models.generate_content(model=MODEL, contents="Hi")
    response = make_client(fake).models.generate_content(model=MODEL, contents="Hi")
    assert_type(response, google_types.GenerateContentResponse)
    assert type(response) is type(expected)
    assert response.text == "answer #2"
    assert not caelum.cache_hit(response)


def test_other_methods_are_forwarded(fake: FakeGemini) -> None:
    client = make_client(fake)
    assert client.models.generate_content_stream.__func__ is (  # type: ignore[attr-defined]
        google_genai.models.Models.generate_content_stream
    )
    assert client.vertexai is False
    assert "generate_content_stream" in dir(client.models)


def test_positional_arguments_reach_the_sdk_unchanged(fake: FakeGemini) -> None:
    with pytest.raises(TypeError):
        make_client(fake).models.generate_content(MODEL, "Hi")  # type: ignore[misc]


# -- cache -------------------------------------------------------------------------


def test_exact_cache_hit_and_metrics(fake: FakeGemini, cache_path: Path) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    first = ask(client)
    second = ask(client)
    assert fake.calls == 1
    assert second.text == first.text == "answer #1"
    assert type(second) is google_types.GenerateContentResponse
    assert caelum.cache_hit(second) and caelum.cache_type(second) == "exact"
    summary = client.metrics.summary()
    assert summary["cache_hits"] == 1
    assert summary["cost_usd"] == pytest.approx(CALL_COST)
    assert summary["cost_saved_usd"] == pytest.approx(CALL_COST)


def test_different_requests_do_not_share_answers(fake: FakeGemini, cache_path: Path) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    ask(client, contents="Question A")
    ask(client, contents="Question B")
    ask(client, contents="Question A", config=google_types.GenerateContentConfig(temperature=0))
    assert fake.calls == 3


def test_config_object_and_dict_share_the_cache(fake: FakeGemini, cache_path: Path) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    ask(client, config=google_types.GenerateContentConfig(temperature=0.2))
    hit = ask(client, config={"temperature": 0.2})
    assert fake.calls == 1
    assert caelum.cache_hit(hit)


def test_python_function_tools_are_never_cached(fake: FakeGemini, cache_path: Path) -> None:
    def get_weather(city: str) -> str:
        """Weather in a city."""
        return "sunny"

    client = make_client(fake, cache=True, cache_path=cache_path)
    config = google_types.GenerateContentConfig(tools=[get_weather])
    ask(client, config=config)
    ask(client, config=config)
    assert fake.calls == 2


def test_pydantic_response_schema_is_not_cached_and_does_not_crash(
    fake: FakeGemini, cache_path: Path
) -> None:
    class Answer(BaseModel):
        text: str

    client = make_client(fake, cache=True, cache_path=cache_path)
    config: dict[str, Any] = {"response_mime_type": "application/json", "response_schema": Answer}
    ask(client, config=config)
    ask(client, config=config)
    assert fake.calls == 2


def test_cache_disabled_by_default(fake: FakeGemini) -> None:
    client = make_client(fake)
    ask(client)
    ask(client)
    assert fake.calls == 2


# -- async ---------------------------------------------------------------------------


async def test_async_shares_cache_and_metrics(fake: FakeGemini, cache_path: Path) -> None:
    client = make_client(fake, cache=True, cache_path=cache_path)
    first = await client.aio.models.generate_content(model=MODEL, contents="Hi")
    assert_type(first, google_types.GenerateContentResponse)
    ask(client, contents="Hi")
    again = await client.aio.models.generate_content(model=MODEL, contents="Hi")
    assert fake.calls == 1
    assert caelum.cache_hit(again)
    assert client.metrics.summary()["cache_hits"] == 2


# -- retries -------------------------------------------------------------------------


def test_retry_on_rate_limit(fake: FakeGemini) -> None:
    fake.failures = [429, 503]
    client = make_client(fake, retry=FAST)
    assert ask(client).text == "answer #1"
    assert client.metrics.summary()["retries"] == 2


async def test_async_retry(fake: FakeGemini) -> None:
    fake.failures = [500]
    client = make_client(fake, retry=FAST)
    response = await client.aio.models.generate_content(model=MODEL, contents="Hi")
    assert response.text == "answer #1"


def test_client_errors_are_not_retried_and_keep_sdk_type(fake: FakeGemini) -> None:
    fake.failures = [400]
    client = make_client(fake, retry=FAST)
    with pytest.raises(google_errors.ClientError) as info:
        ask(client)
    assert info.value.code == 400
    assert fake.calls == 0


def test_genai_errors_are_classified_by_code() -> None:
    assert is_retryable(google_errors.ClientError(429, {}), FAST)
    assert is_retryable(google_errors.ServerError(503, {}), FAST)
    assert not is_retryable(google_errors.ClientError(404, {}), FAST)
    assert is_retryable(httpx.ConnectError("down"), FAST)


# -- usage ---------------------------------------------------------------------------


def test_usage_counts_thinking_tokens_as_output() -> None:
    response = google_types.GenerateContentResponse(
        usage_metadata=google_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=4, thoughts_token_count=6
        )
    )
    assert extract_usage(response) == (10, 10)
    assert extract_usage(google_types.GenerateContentResponse()) is None
