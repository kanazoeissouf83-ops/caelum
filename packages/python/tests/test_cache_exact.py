"""Exact cache: identical requests are served from the local cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import openai
import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

import caelum
from caelum import AsyncOpenAI, OpenAI
from caelum.cache import CacheStore, SQLiteStore, compute_cache_key, should_cache
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "How do I return an item?"}]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class BrokenStore(CacheStore):
    """A store whose every operation fails (corrupted file, full disk...)."""

    def get(self, key: str) -> Any | None:
        raise OSError("disk on fire")

    def set(self, key: str, value: Any, ttl: float | None) -> None:
        raise OSError("disk on fire")

    def delete(self, key: str) -> None:
        raise OSError("disk on fire")

    def clear(self) -> None:
        raise OSError("disk on fire")

    def close(self) -> None:
        pass


def client(server: FakeServer, cache_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {"cache": True, "cache_path": cache_path, **kwargs}
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: OpenAI, **kwargs: Any) -> ChatCompletion:
    params: dict[str, Any] = {"model": "gpt-4o", "messages": MESSAGES, **kwargs}
    return cast(ChatCompletion, c.chat.completions.create(**params))


def content(response: ChatCompletion) -> str | None:
    return response.choices[0].message.content


# -- core behaviour ------------------------------------------------------------


def test_identical_request_is_a_hit(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    first = ask(c)
    second = ask(c)
    assert server.calls == 1
    assert content(first) == content(second) == "answer #1"
    assert isinstance(second, ChatCompletion)
    assert not caelum.cache_hit(first)
    assert caelum.cache_hit(second)
    assert caelum.cache_type(second) == "exact"


VARIANTS: list[dict[str, Any]] = [
    {"temperature": 0.9},
    {"model": "gpt-4o-mini"},
    {"messages": [{"role": "user", "content": "Something else"}]},
    {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
    {"response_format": {"type": "json_object"}},
    {"seed": 42},
]


@pytest.mark.parametrize("variant", VARIANTS)
def test_any_answer_changing_parameter_is_a_miss(
    server: FakeServer, cache_path: Path, variant: dict[str, Any]
) -> None:
    c = client(server, cache_path)
    ask(c)
    response = ask(c, **variant)
    assert server.calls == 2
    assert not caelum.cache_hit(response)


def test_transport_parameters_do_not_affect_the_key(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    ask(c)
    response = ask(c, timeout=30, extra_headers={"X-Trace": "abc"}, user="u-1", metadata={"a": "b"})
    assert server.calls == 1
    assert caelum.cache_hit(response)


def test_cache_is_disabled_by_default(server: FakeServer, tmp_path: Path) -> None:
    c = OpenAI(api_key="sk-test", http_client=server.http_client())
    ask(c)
    second = ask(c)
    assert server.calls == 2
    assert not caelum.cache_hit(second)
    assert not (tmp_path / "caelum-home").exists()  # nothing written to disk


def test_explicit_cache_false(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path, cache=False)
    ask(c)
    ask(c)
    assert server.calls == 2
    assert not cache_path.exists()


def test_default_cache_location_honours_caelum_home(server: FakeServer, tmp_path: Path) -> None:
    c = OpenAI(api_key="sk-test", http_client=server.http_client(), cache=True)
    ask(c)
    assert (tmp_path / "caelum-home" / "cache.sqlite").exists()


def test_entries_expire_after_ttl(server: FakeServer, cache_path: Path) -> None:
    clock = FakeClock()
    store = SQLiteStore(cache_path, clock=clock)
    c = client(server, cache_path, cache_store=store, cache_ttl=60)
    ask(c)
    clock.now += 59
    assert caelum.cache_hit(ask(c))
    clock.now += 2
    assert not caelum.cache_hit(ask(c))
    assert server.calls == 2


def test_ttl_none_means_the_maximum_of_30_days(server: FakeServer, cache_path: Path) -> None:
    clock = FakeClock()
    store = SQLiteStore(cache_path, clock=clock)
    c = client(server, cache_path, cache_store=store, cache_ttl=None)
    ask(c)
    clock.now += 29 * 86_400
    assert caelum.cache_hit(ask(c))
    clock.now += 2 * 86_400
    assert not caelum.cache_hit(ask(c))


def test_streams_are_never_cached(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    for _ in range(2):
        stream = c.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
        assert "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert server.calls == 2


def test_tool_choice_required_is_not_cached(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    tools: list[Any] = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    ask(c, tools=tools, tool_choice="required")
    ask(c, tools=tools, tool_choice="required")
    assert server.calls == 2


def test_embeddings_are_cached(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    c.embeddings.create(model="text-embedding-3-small", input="hello")
    response = c.embeddings.create(model="text-embedding-3-small", input="hello")
    assert isinstance(response, CreateEmbeddingResponse)
    assert caelum.cache_hit(response)
    assert server.calls == 1


def test_cache_persists_across_clients(server: FakeServer, cache_path: Path) -> None:
    ask(client(server, cache_path))
    assert caelum.cache_hit(ask(client(server, cache_path)))
    assert server.calls == 1


def test_different_base_urls_never_share_answers(server: FakeServer, cache_path: Path) -> None:
    ask(client(server, cache_path))
    other = client(server, cache_path, base_url="https://my-proxy.example.com/v1")
    assert not caelum.cache_hit(ask(other))
    assert server.calls == 2


def test_different_api_keys_never_share_answers(server: FakeServer, cache_path: Path) -> None:
    """Multi-tenant safety: one customer's key must never get another's answers."""
    ask(client(server, cache_path))
    other = OpenAI(
        api_key="sk-other-tenant",
        http_client=server.http_client(),
        cache=True,
        cache_path=cache_path,
    )
    assert not caelum.cache_hit(ask(other))
    assert caelum.cache_hit(ask(other))
    assert server.calls == 2


def test_api_key_is_never_written_to_the_cache_file(server: FakeServer, cache_path: Path) -> None:
    store = SQLiteStore(cache_path)
    ask(client(server, cache_path, cache_store=store))
    store.flush()
    assert b"sk-test" not in cache_path.read_bytes()


def test_copies_share_the_cache(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    ask(c)
    assert caelum.cache_hit(ask(c.with_options(timeout=10)))


def test_each_hit_returns_a_fresh_object(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    ask(c)
    hit = ask(c)
    hit.choices[0].message.content = "mutated by the caller"
    assert content(ask(c)) == "answer #1"


# -- robustness: CAELUM never breaks a working request -------------------------


def test_broken_store_falls_back_to_the_api(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path, cache_store=BrokenStore())
    assert content(ask(c)) == "answer #1"
    assert content(ask(c)) == "answer #2"


def test_corrupted_cache_file_disables_cache_without_crashing(
    server: FakeServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = tmp_path / "garbage.sqlite"
    bad.write_text("this is not a database " * 100)
    c = client(server, bad)
    assert content(ask(c)) == "answer #1"
    assert content(ask(c)) == "answer #2"
    assert "cache disabled" in caplog.text


def test_directory_as_cache_path_disables_cache_without_crashing(
    server: FakeServer, tmp_path: Path
) -> None:
    c = client(server, tmp_path)
    assert content(ask(c)) == "answer #1"


def test_corrupted_file_with_semantic_cache_does_not_crash(
    server: FakeServer, tmp_path: Path
) -> None:
    bad = tmp_path / "garbage.sqlite"
    bad.write_text("this is not a database " * 100)
    c = client(server, bad, semantic_cache=True)
    assert content(ask(c)) == "answer #1"


def test_unreadable_entry_is_dropped(server: FakeServer, cache_path: Path) -> None:
    store = SQLiteStore(cache_path)
    c = client(server, cache_path, cache_store=store)
    ask(c)
    store.flush()
    rows = store._conn.execute("SELECT key FROM caelum_cache").fetchall()  # pyright: ignore[reportPrivateUsage]
    (key,) = (str(row[0]) for row in rows)
    store.set(key, {"not": "a chat completion"}, None)
    response = ask(c)
    assert not caelum.cache_hit(response)
    assert server.calls == 2
    assert caelum.cache_hit(ask(c))  # the fresh answer replaced the bad entry


def test_unhashable_request_gets_no_key() -> None:
    request: dict[str, Any] = {"model": "m", "messages": MESSAGES, "temperature": float("nan")}
    assert compute_cache_key(request, provider="openai", endpoint="chat", namespace="ns") is None
    assert compute_cache_key({"x": object()}, provider="o", endpoint="c", namespace="n") is None


def test_api_errors_are_not_cached(server: FakeServer, cache_path: Path) -> None:
    server.failures = [400]
    c = client(server, cache_path)
    with pytest.raises(openai.BadRequestError):
        ask(c)
    assert content(ask(c)) == "answer #1"


# -- configuration validation --------------------------------------------------


def test_invalid_ttl_is_rejected(cache_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_ttl"):
        OpenAI(api_key="sk-test", cache=True, cache_path=cache_path, cache_ttl=0)


def test_non_boolean_cache_flag_is_rejected() -> None:
    with pytest.raises(TypeError):
        OpenAI(api_key="sk-test", cache="yes")  # pyright: ignore[reportArgumentType]


# -- async -------------------------------------------------------------------------


async def test_async_hit(server: FakeServer, cache_path: Path) -> None:
    c = AsyncOpenAI(
        api_key="sk-test", http_client=server.async_http_client(), cache=True, cache_path=cache_path
    )
    await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    second = await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert caelum.cache_hit(second)
    assert server.calls == 1


# -- key computation ---------------------------------------------------------------


def test_key_is_deterministic_and_ignores_sentinels() -> None:
    a: dict[str, Any] = {
        "model": "m",
        "messages": MESSAGES,
        "temperature": 0.5,
        "top_p": openai.NOT_GIVEN,
    }
    b: dict[str, Any] = {"temperature": 0.5, "messages": MESSAGES, "model": "m"}
    key_a = compute_cache_key(a, provider="openai", endpoint="chat", namespace="ns")
    assert key_a is not None and len(key_a) == 64
    assert key_a == compute_cache_key(b, provider="openai", endpoint="chat", namespace="ns")
    assert key_a != compute_cache_key(b, provider="openai", endpoint="chat", namespace="other")
    with_stream_false = {**b, "stream": False}
    assert key_a == compute_cache_key(
        with_stream_false, provider="openai", endpoint="chat", namespace="ns"
    )


def test_should_cache_rules() -> None:
    assert should_cache({"model": "m"})
    assert not should_cache({"model": "m", "stream": True})
    assert not should_cache({"model": "m", "tool_choice": "required"})


# -- SQLite store ------------------------------------------------------------------


def test_sqlite_store_operations(cache_path: Path) -> None:
    store = SQLiteStore(cache_path)
    store.set("k", {"v": 1}, None)
    assert store.get("k") == {"v": 1}
    assert "k" in store
    assert "missing" not in store
    assert 42 not in store
    assert len(store) == 1
    store.delete("k")
    assert store.get("k") is None
    store.set("a", [1], None)
    store.clear()
    assert len(store) == 0
    store.close()


def test_sqlite_store_purge_expired(cache_path: Path) -> None:
    clock = FakeClock()
    store = SQLiteStore(cache_path, clock=clock)
    store.set("short", 1, 5)
    store.set("long", 2, 500)
    clock.now += 10
    assert store.purge_expired() == 1
    assert store.get("long") == 2


def test_sqlite_in_memory_store() -> None:
    store = SQLiteStore(":memory:")
    store.set("k", "v", 60)
    assert store.get("k") == "v"
