"""Local backends (Ollama, llama.cpp, vLLM, LM Studio): free fallback and routing targets."""

from __future__ import annotations

from typing import Any

import anthropic
import openai
import pytest
from openai.types.chat import ChatCompletion

import caelum
from caelum import Anthropic, AsyncOpenAI, OpenAI
from caelum.fallback import ProviderPool
from caelum.providers import detect_provider, dialect, is_local, local_base_url
from caelum.translate import translate_request
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "Where is my order?"}]
OLLAMA_URL = "http://localhost:11434/v1"


def raw_ollama(server: FakeServer) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=OLLAMA_URL, api_key="local", http_client=server.http_client(), max_retries=0
    )


def cloud_client(server: FakeServer, fallback: list[Any], **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "fallback": fallback,
        "provider_clients": {"ollama": raw_ollama(server)},
        "max_retries": 0,
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, **kwargs: Any) -> Any:
    params: dict[str, Any] = {"model": "gpt-4o", "messages": MESSAGES, **kwargs}
    return c.chat.completions.create(**params)


# -- provider helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434/v1", "ollama"),
        ("http://127.0.0.1:11434/v1/", "ollama"),
        ("http://localhost:8080/v1", "llamacpp"),
        ("http://localhost:8000/v1", "vllm"),
        ("http://localhost:1234/v1", "lmstudio"),
        ("http://[::1]:11434/v1", "ollama"),
        ("http://localhost:5000/v1", "local"),
        ("https://api.openai.com/v1", "openai"),
        ("https://my-proxy.example.com:11434/v1", "openai"),
        ("not a url", "openai"),
    ],
)
def test_detect_provider(url: str, expected: str) -> None:
    assert detect_provider(url) == expected


def test_local_providers_speak_openai_and_cost_nothing() -> None:
    for name in ("ollama", "llamacpp", "vllm", "lmstudio", "local"):
        assert dialect(name) == "openai"
        assert is_local(name)
    assert dialect("anthropic") == "anthropic"
    assert not is_local("openai")


def test_local_base_url_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("CAELUM_OLLAMA_BASE_URL", raising=False)
    assert local_base_url("ollama") == OLLAMA_URL
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:9999")
    assert local_base_url("ollama") == "http://localhost:9999/v1"
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434/")
    assert local_base_url("ollama") == "http://gpu-box:11434/v1"
    monkeypatch.setenv("CAELUM_OLLAMA_BASE_URL", "http://other:1/v1")
    assert local_base_url("ollama") == "http://other:1/v1"
    monkeypatch.setenv("CAELUM_VLLM_BASE_URL", "http://vllm-server:8000/v1")
    assert local_base_url("vllm") == "http://vllm-server:8000/v1"


def test_pool_builds_openai_clients_for_local_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAELUM_LLAMACPP_BASE_URL", raising=False)
    pool = ProviderPool(native="anthropic", is_async=False)
    pool.require(["ollama", "llamacpp"])
    raw = pool.get("llamacpp")
    assert isinstance(raw, openai.OpenAI)
    assert str(raw.base_url).startswith("http://localhost:8080/v1")
    async_raw = ProviderPool(native="openai", is_async=True).get("ollama")
    assert isinstance(async_raw, openai.AsyncOpenAI)


def test_anthropic_requests_translate_to_local_backends() -> None:
    request = {"model": "qwen3:1.7b", "max_tokens": 10, "messages": MESSAGES}
    translated = translate_request("anthropic", "ollama", request)
    assert translated["messages"] == MESSAGES
    assert translate_request("ollama", "openai", {"model": "x"}) == {"model": "x"}


# -- cloud first, local as the last resort ------------------------------------------


def test_cloud_outage_falls_back_to_ollama_for_free(server: FakeServer) -> None:
    server.failures = [503]
    c = cloud_client(server, ["ollama:qwen3:1.7b"])
    response = ask(c)
    assert isinstance(response, ChatCompletion)
    assert caelum.routed_to(response) == "qwen3:1.7b"
    assert server.requests[-1]["port"] == 11434
    assert server.requests[-1]["body"]["model"] == "qwen3:1.7b"
    s = c.metrics.summary()
    assert s["fallbacks"] == 1
    assert s["cost_usd"] == 0.0


def test_streaming_can_fall_back_to_a_local_backend(server: FakeServer) -> None:
    server.failures = [503]
    c = cloud_client(server, ["ollama:qwen3:1.7b"])
    stream = ask(c, stream=True)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "answer #1"
    assert server.requests[-1]["port"] == 11434


def test_local_backend_down_too_raises_the_original_error(server: FakeServer) -> None:
    server.failures = [503, 503]
    with pytest.raises(openai.InternalServerError):
        ask(cloud_client(server, ["ollama:qwen3:1.7b"]))


def test_anthropic_client_falls_back_to_ollama(server: FakeServer) -> None:
    server.failures = [529]
    c = Anthropic(
        api_key="sk-ant-test",
        http_client=server.http_client(),
        max_retries=0,
        fallback=["ollama:qwen3:1.7b"],
        provider_clients={"ollama": raw_ollama(server)},
    )
    response: Any = c.messages.create(model="claude-haiku-5-5", max_tokens=50, messages=MESSAGES)
    assert isinstance(response, anthropic.types.Message)
    assert response.content[0].type == "text"
    assert server.requests[-1]["port"] == 11434
    assert c.metrics.summary()["cost_usd"] == 0.0


async def test_async_cloud_outage_falls_back_to_ollama(server: FakeServer) -> None:
    server.failures = [502]
    raw = openai.AsyncOpenAI(
        base_url=OLLAMA_URL,
        api_key="local",
        http_client=server.async_http_client(),
        max_retries=0,
    )
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        max_retries=0,
        fallback=["ollama:qwen3:1.7b"],
        provider_clients={"ollama": raw},
    )
    response = await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert caelum.routed_to(response) == "qwen3:1.7b"
    assert server.requests[-1]["port"] == 11434


# -- local first, cloud as the backup -----------------------------------------------


def local_client(server: FakeServer, **kwargs: Any) -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_URL,
        api_key="local",
        http_client=server.http_client(),
        max_retries=0,
        **kwargs,
    )


def test_local_primary_falls_back_to_the_real_openai(server: FakeServer) -> None:
    server.failures = [503]
    real_openai = openai.OpenAI(api_key="sk-test", http_client=server.http_client(), max_retries=0)
    c = local_client(
        server, fallback=["openai:gpt-4o-mini"], provider_clients={"openai": real_openai}
    )
    response = ask(c, model="qwen3:1.7b")
    assert caelum.routed_to(response) == "gpt-4o-mini"
    # The "openai" tier is OpenAI itself, not the local server the client points to.
    assert server.requests[0]["port"] == 11434
    assert server.requests[-1]["port"] is None


def test_local_primary_requests_are_free(server: FakeServer) -> None:
    c = local_client(server)
    ask(c, model="gpt-4o")  # even under a cloud model name, a local answer costs nothing
    assert c.metrics.summary()["cost_usd"] == 0.0


def test_local_primary_cache_hits_save_nothing(server: FakeServer, cache_path: Any) -> None:
    c = local_client(server, cache=True, cache_path=cache_path)
    ask(c, model="gpt-4o")
    assert caelum.cache_hit(ask(c, model="gpt-4o"))
    assert c.metrics.summary()["cost_saved_usd"] == 0.0


def test_router_can_send_small_requests_to_a_local_model(server: FakeServer) -> None:
    config = {
        "routes": [
            {
                "name": "small-local",
                "match": {"messages_max_tokens": 50},
                "target": "qwen3:1.7b",
                "provider": "ollama",
            }
        ]
    }
    c = cloud_client(server, ["anthropic:claude-haiku-5-5"], router_config=config)
    response = ask(c)
    assert caelum.routed_to(response) == "qwen3:1.7b"
    assert server.requests[-1]["port"] == 11434
    assert c.metrics.summary()["cost_usd"] == 0.0


def test_unknown_provider_names_are_still_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported fallback provider"):
        OpenAI(api_key="sk-test", fallback=["llama:qwen3"])
