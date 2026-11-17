"""The OpenAI wrapper must behave exactly like ``openai.OpenAI`` (drop-in guarantee)."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import openai
import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion
from typing_extensions import assert_type

import caelum
from caelum import AsyncOpenAI, OpenAI
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "How do I return an item?"}]


_INTERNAL_CLIENT = "_caelum_client"
MISSING = "does_not_exist"


def raw(client: object) -> Any:
    """Return the wrapped SDK client (internal attribute)."""
    return getattr(client, _INTERNAL_CLIENT)


def make_client(server: FakeServer, **kwargs: Any) -> OpenAI:
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **kwargs)


def make_async_client(server: FakeServer, **kwargs: Any) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="sk-test", http_client=server.async_http_client(), **kwargs)


# -- public API ---------------------------------------------------------------


def test_public_exports() -> None:
    from caelum.openai import AsyncCaelumOpenAI, CaelumOpenAI

    assert caelum.OpenAI is CaelumOpenAI
    assert caelum.AsyncOpenAI is AsyncCaelumOpenAI
    with pytest.raises(AttributeError):
        _ = caelum.DoesNotExist  # pyright: ignore[reportAttributeAccessIssue]


def test_missing_openai_extra_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)  # simulate "not installed"
    monkeypatch.delitem(sys.modules, "caelum.openai", raising=False)
    with pytest.raises(ImportError, match=r'pip install "caelum-ai\[openai\]"'):
        importlib.import_module("caelum.openai")


# -- sync behaviour -------------------------------------------------------------


def test_chat_completion_is_identical_to_raw_sdk(server: FakeServer) -> None:
    raw = openai.OpenAI(api_key="sk-test", http_client=server.http_client())
    expected = raw.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    response = make_client(server).chat.completions.create(model="gpt-4o", messages=MESSAGES)

    assert isinstance(response, ChatCompletion)
    assert type(response) is type(expected)
    assert response.model == expected.model
    assert response.choices[0].message.content == "answer #2"
    assert not caelum.cache_hit(response)
    assert caelum.cache_type(response) is None


def test_arguments_are_forwarded_untouched(server: FakeServer) -> None:
    make_client(server).chat.completions.create(
        model="gpt-4o", messages=MESSAGES, temperature=0.2, max_tokens=50, top_p=0.9, seed=7
    )
    body = server.requests[-1]["body"]
    assert body == {
        "model": "gpt-4o",
        "messages": MESSAGES,
        "temperature": 0.2,
        "max_tokens": 50,
        "top_p": 0.9,
        "seed": 7,
    }


def test_embeddings_create(server: FakeServer) -> None:
    response = make_client(server).embeddings.create(
        model="text-embedding-3-small", input=["a", "b"]
    )
    assert isinstance(response, CreateEmbeddingResponse)
    assert len(response.data) == 2
    assert server.requests[-1]["path"].endswith("/embeddings")


def test_streaming_works(server: FakeServer) -> None:
    stream = make_client(server).chat.completions.create(
        model="gpt-4o", messages=MESSAGES, stream=True
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert text == "answer #1"


def test_api_errors_keep_sdk_exception_types(server: FakeServer) -> None:
    server.failures = [400]
    with pytest.raises(openai.BadRequestError):
        make_client(server).chat.completions.create(model="gpt-4o", messages=MESSAGES)


def test_positional_arguments_fail_like_the_sdk(server: FakeServer) -> None:
    client = make_client(server)
    with pytest.raises(TypeError):
        client.chat.completions.create("gpt-4o")  # pyright: ignore[reportCallIssue]


# -- attribute forwarding (__getattr__ / __setattr__) ---------------------------


def test_attributes_are_forwarded(server: FakeServer) -> None:
    client = make_client(server, max_retries=5)
    assert client.api_key == "sk-test"
    assert client.max_retries == 5
    assert str(client.base_url) == "https://api.openai.com/v1/"
    assert client.models is raw(client).models
    assert hasattr(client.chat.completions, "with_raw_response")  # non-intercepted method


def test_setting_attributes_updates_the_real_client(server: FakeServer) -> None:
    client = make_client(server)
    client.api_key = "sk-rotated"
    assert raw(client).api_key == "sk-rotated"


def test_unknown_attribute_raises_attribute_error(server: FakeServer) -> None:
    client = make_client(server)
    with pytest.raises(AttributeError):
        getattr(client, MISSING)
    with pytest.raises(AttributeError):
        getattr(client.chat.completions, MISSING)


def test_dir_lists_sdk_attributes(server: FakeServer) -> None:
    client = make_client(server)
    assert {"chat", "embeddings", "models", "api_key"} <= set(dir(client))


def test_copy_and_with_options_stay_wrapped(server: FakeServer) -> None:
    client = make_client(server)
    for derived in (client.copy(max_retries=1), client.with_options(timeout=5)):
        assert isinstance(derived, OpenAI)
        response = derived.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert not caelum.cache_hit(response)
    assert client.copy(max_retries=1).max_retries == 1


def test_context_manager_closes_client(server: FakeServer) -> None:
    with make_client(server) as client:
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert raw(client).is_closed()


def test_repr_mentions_caelum(server: FakeServer) -> None:
    assert "caelum.CaelumOpenAI" in repr(make_client(server))


# -- async behaviour ------------------------------------------------------------


async def test_async_chat_completion(server: FakeServer) -> None:
    async with make_async_client(server) as client:
        response = await client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "answer #1"


async def test_async_embeddings(server: FakeServer) -> None:
    client = make_async_client(server)
    response = await client.embeddings.create(model="text-embedding-3-small", input="a")
    assert isinstance(response, CreateEmbeddingResponse)


async def test_async_streaming(server: FakeServer) -> None:
    client = make_async_client(server)
    stream = await client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
    parts = [chunk.choices[0].delta.content or "" async for chunk in stream]
    assert "".join(parts) == "answer #1"


async def test_async_copy_stays_wrapped(server: FakeServer) -> None:
    derived = make_async_client(server).with_options(max_retries=0)
    assert isinstance(derived, AsyncOpenAI)


def test_static_types_match_the_sdk(server: FakeServer) -> None:
    """Checked by pyright: users keep the SDK's precise return types."""
    client = make_client(server)
    response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert_type(response, ChatCompletion)
    embeddings = client.embeddings.create(model="text-embedding-3-small", input="a")
    assert_type(embeddings, CreateEmbeddingResponse)
