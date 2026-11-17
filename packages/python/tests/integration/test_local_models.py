"""Real checks against a local Ollama server (free: no key, no per-token cost).

Needs Ollama running (https://ollama.com) with the test model pulled::

    ollama pull gemma3:1b
    uv run pytest -m integration -k local

The model can be changed with ``CAELUM_IT_OLLAMA_MODEL``. Skipped when Ollama is not
reachable or the model is missing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import caelum
from caelum import Anthropic, AsyncOpenAI, OpenAI
from caelum.providers import local_base_url
from tests.integration.conftest import ANTHROPIC_MODEL, OPENAI_MODEL, record, user

pytestmark = pytest.mark.integration

OLLAMA_MODEL = os.environ.get("CAELUM_IT_OLLAMA_MODEL", "gemma3:1b")
UNREACHABLE = "http://127.0.0.1:9/v1"  # nothing listens here: connection error
LOCAL_MAX_TOKENS = 32


@pytest.fixture(scope="module")
def ollama() -> str:
    """The test model name, once Ollama is known to serve it (else skip)."""
    base = local_base_url("ollama").removesuffix("/v1")
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        pytest.skip(f"Ollama not reachable at {base}")
    names = {model["name"] for model in tags.get("models", [])}
    if OLLAMA_MODEL not in names and f"{OLLAMA_MODEL}:latest" not in names:
        pytest.skip(f"Ollama model {OLLAMA_MODEL} not pulled (ollama pull {OLLAMA_MODEL})")
    return OLLAMA_MODEL


def text_of(response: Any) -> str:
    return response.choices[0].message.content or ""


def test_ollama_as_primary_with_cache(ollama: str, tmp_path: Path) -> None:
    client = OpenAI(
        base_url=local_base_url("ollama"),
        api_key="ollama",
        cache=True,
        cache_path=tmp_path / "c.sqlite",
    )
    questions = [f"Reply with the number {i} only." for i in range(3)]
    for question in questions * 2:  # 6 calls, 3 duplicates
        response = client.chat.completions.create(
            model=ollama, messages=user(question), max_tokens=LOCAL_MAX_TOKENS
        )
        assert text_of(response)
    s = client.metrics.summary()
    record(
        f"Local: Ollama {ollama} as primary + cache (6 calls)",
        f"{s['cache_hits']} hits, cost ${s['cost_usd']:.2f}, p95 {s['latency_p95_ms']:.0f} ms",
    )
    assert s["cache_hits"] == 3
    assert s["cost_usd"] == 0.0


def test_cloud_down_falls_back_to_ollama(ollama: str) -> None:
    client = OpenAI(
        api_key="sk-unused",
        base_url="https://api.openai.com.invalid/v1",  # DNS failure: the cloud is "down"
        max_retries=0,
        timeout=5,
        fallback=[f"ollama:{ollama}"],
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=user("Say ok."), max_tokens=LOCAL_MAX_TOKENS
    )
    elapsed = time.perf_counter() - started
    record(
        "Local: OpenAI unreachable -> Ollama fallback",
        f"answered by {caelum.routed_to(response)} in {elapsed:.1f} s, cost $0",
    )
    assert caelum.routed_to(response) == ollama
    assert text_of(response)
    assert client.metrics.summary()["cost_usd"] == 0.0


def test_cloud_down_streams_from_ollama(ollama: str) -> None:
    client = OpenAI(
        api_key="sk-unused",
        base_url="https://api.openai.com.invalid/v1",
        max_retries=0,
        timeout=5,
        fallback=[f"ollama:{ollama}"],
    )
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=user("Count from 1 to 5."),
        max_tokens=LOCAL_MAX_TOKENS,
        stream=True,
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    record("Local: streaming fallback to Ollama", f"{len(text)} characters")
    assert text


def test_anthropic_down_falls_back_to_ollama(ollama: str) -> None:
    client = Anthropic(
        api_key="sk-ant-unused",
        base_url="https://api.anthropic.com.invalid",
        max_retries=0,
        timeout=5,
        fallback=[f"ollama:{ollama}"],
    )
    message: Any = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=LOCAL_MAX_TOKENS, messages=user("Say ok.")
    )
    record("Local: Anthropic unreachable -> Ollama (translated)", f"stop={message.stop_reason}")
    assert message.content[0].text


async def test_async_cloud_down_falls_back_to_ollama(ollama: str) -> None:
    client = AsyncOpenAI(
        api_key="sk-unused",
        base_url="https://api.openai.com.invalid/v1",
        max_retries=0,
        timeout=5,
        fallback=[f"ollama:{ollama}"],
    )
    response = await client.chat.completions.create(
        model=OPENAI_MODEL, messages=user("Say ok."), max_tokens=LOCAL_MAX_TOKENS
    )
    record("Local: async fallback to Ollama", f"answered by {caelum.routed_to(response)}")
    assert caelum.routed_to(response) == ollama


def test_local_server_down_falls_back_to_openai(openai_key: str) -> None:
    client = OpenAI(
        base_url="http://localhost:11435/v1",  # a local server that is not running
        api_key="local",
        max_retries=0,
        timeout=5,
        fallback=[f"openai:{OPENAI_MODEL}"],
        provider_keys={"openai": openai_key},
    )
    response = client.chat.completions.create(
        model="any-local-model", messages=user("Say ok."), max_tokens=16
    )
    record(
        "Local: local server down -> real OpenAI fallback",
        f"answered by {caelum.routed_to(response)}",
    )
    assert caelum.routed_to(response) == OPENAI_MODEL
    assert text_of(response)
