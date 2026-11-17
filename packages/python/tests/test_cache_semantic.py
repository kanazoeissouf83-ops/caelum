"""Semantic cache: similar questions in the same context reuse answers — and nothing else."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import openai
import pytest
from openai.types.chat import ChatCompletion

import caelum
from caelum import Anthropic, AsyncOpenAI, OpenAI, OpenAIEmbedder
from caelum.cache import EmbeddingResult, SQLiteVectorStore
from caelum.cache.semantic import anthropic_semantic_query, openai_semantic_query
from caelum.cache.vectors import normalize
from tests.conftest import FakeServer

FIXTURES = Path(__file__).parent / "fixtures"
STATE_ATTR = "_caelum_state"

RETURN_ITEM = "How do I return an item?"
SEND_BACK = "How can I send back a product?"  # paraphrase: cosine ~0.990
SHOES_WINDOW = "What is the return window for shoes?"  # related but different: ~0.90

VECTORS: dict[str, list[float]] = {
    RETURN_ITEM: [1.0, 0.0, 0.0, 0.0],
    SEND_BACK: [0.99, 0.141, 0.0, 0.0],
    SHOES_WINDOW: [0.9, 0.436, 0.0, 0.0],
}


def _pseudo_random_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(byte / 255) * 2 - 1 for byte in digest[:8]]


class FakeEmbedder:
    """Deterministic embedder with hand-picked similarities."""

    def __init__(self, model: str = "fake-embedding-model") -> None:
        self._model = model
        self.calls: list[str] = []
        self.fail = False

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("embedding API down")
        vector = VECTORS.get(text) or _pseudo_random_vector(text)
        padded = vector + [0.0] * (8 - len(vector))
        return EmbeddingResult(vector=padded, tokens=len(text.split()))


class AsyncOnlyEmbedder:
    """Embedder exposing only ``aembed`` (no synchronous ``embed``)."""

    @property
    def model(self) -> str:
        return "async-only"

    async def aembed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[1.0, 0.0], tokens=1)


def client(server: FakeServer, cache_path: Path, embedder: Any = None, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "semantic_cache": True,
        "semantic_embedder": embedder if embedder is not None else FakeEmbedder(),
        "cache_path": cache_path,
        # Embedding-only mode by default here; verifier tests opt back in.
        "semantic_verifier": False,
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: OpenAI, question: str, system: str | None = None, **kwargs: Any) -> ChatCompletion:
    messages: list[Any] = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": question})
    params: dict[str, Any] = {"model": "gpt-4o", "messages": messages, **kwargs}
    return cast(ChatCompletion, c.chat.completions.create(**params))


def content(response: ChatCompletion) -> str | None:
    return response.choices[0].message.content


# -- core behaviour ----------------------------------------------------------------


def test_paraphrase_is_a_semantic_hit(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    first = ask(c, RETURN_ITEM)
    second = ask(c, SEND_BACK)
    assert server.calls == 1
    assert content(second) == content(first) == "answer #1"
    assert caelum.cache_hit(second)
    assert caelum.cache_type(second) == "semantic"
    s = c.metrics.summary()
    assert s["cache_hits_semantic"] == 1
    assert s["cost_saved_usd"] > 0


def test_related_but_different_question_is_a_miss(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    ask(c, RETURN_ITEM)
    response = ask(c, SHOES_WINDOW)  # cosine ~0.90 < 0.97
    assert not caelum.cache_hit(response)
    assert server.calls == 2


def test_threshold_is_adjustable(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path, semantic_threshold=0.85)
    ask(c, RETURN_ITEM)
    assert caelum.cache_type(ask(c, SHOES_WINDOW)) == "semantic"


def test_semantic_cache_is_disabled_by_default(server: FakeServer, cache_path: Path) -> None:
    embedder = FakeEmbedder()
    c = OpenAI(api_key="sk-test", http_client=server.http_client(), cache_path=cache_path)
    ask(c, RETURN_ITEM)
    ask(c, SEND_BACK)
    assert server.calls == 2
    assert embedder.calls == []
    assert not cache_path.exists()


CONTEXT_VARIANTS: list[dict[str, Any]] = [
    {"system": "Answer in Spanish."},
    {"model": "gpt-4o-mini"},
    {"temperature": 1.5},
    {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
]


@pytest.mark.parametrize("variant", CONTEXT_VARIANTS)
def test_any_context_difference_prevents_reuse(
    server: FakeServer, cache_path: Path, variant: dict[str, Any]
) -> None:
    c = client(server, cache_path)
    ask(c, RETURN_ITEM, system="Answer in French.")
    params = {"system": "Answer in French.", **variant}
    system = params.pop("system")
    response = ask(c, SEND_BACK, system=system, **params)
    assert not caelum.cache_hit(response)


def test_different_history_prevents_reuse(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path)
    history_a: list[Any] = [
        {"role": "user", "content": "I bought shoes."},
        {"role": "assistant", "content": "Great choice!"},
    ]
    history_b: list[Any] = [
        {"role": "user", "content": "I bought a laptop."},
        {"role": "assistant", "content": "Great choice!"},
    ]
    c.chat.completions.create(
        model="gpt-4o", messages=[*history_a, {"role": "user", "content": RETURN_ITEM}]
    )
    response = c.chat.completions.create(
        model="gpt-4o", messages=[*history_b, {"role": "user", "content": SEND_BACK}]
    )
    assert not caelum.cache_hit(response)


def test_semantic_hit_is_promoted_to_the_exact_cache(server: FakeServer, cache_path: Path) -> None:
    embedder = FakeEmbedder()
    c = client(server, cache_path, embedder=embedder, cache=True)
    ask(c, RETURN_ITEM)
    assert caelum.cache_type(ask(c, SEND_BACK)) == "semantic"
    calls_before = len(embedder.calls)
    assert caelum.cache_type(ask(c, SEND_BACK)) == "exact"
    assert len(embedder.calls) == calls_before  # no embedding needed any more


def test_streams_and_multimodal_are_not_semantically_cached(
    server: FakeServer, cache_path: Path
) -> None:
    embedder = FakeEmbedder()
    c = client(server, cache_path, embedder=embedder)
    list(
        c.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": RETURN_ITEM}], stream=True
        )
    )
    image_message: list[Any] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": RETURN_ITEM},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        }
    ]
    c.chat.completions.create(model="gpt-4o", messages=image_message)
    c.chat.completions.create(model="gpt-4o", messages=image_message)
    assert embedder.calls == []
    assert server.calls == 3


def test_embeddings_endpoint_is_never_semantically_cached(
    server: FakeServer, cache_path: Path
) -> None:
    embedder = FakeEmbedder()
    c = client(server, cache_path, embedder=embedder)
    c.embeddings.create(model="text-embedding-3-small", input=RETURN_ITEM)
    c.embeddings.create(model="text-embedding-3-small", input=SEND_BACK)
    assert embedder.calls == []


def test_embedding_failure_falls_back_to_the_api(server: FakeServer, cache_path: Path) -> None:
    embedder = FakeEmbedder()
    embedder.fail = True
    c = client(server, cache_path, embedder=embedder)
    assert content(ask(c, RETURN_ITEM)) == "answer #1"
    assert content(ask(c, SEND_BACK)) == "answer #2"


def test_entries_expire(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path, cache_ttl=60)
    ask(c, RETURN_ITEM)
    state: Any = getattr(c, STATE_ATTR)
    state.semantic_cache.store._clock = lambda: 10**12  # far in the future
    assert not caelum.cache_hit(ask(c, SEND_BACK))


def test_embedding_cost_is_tracked(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache_path, embedder=FakeEmbedder(model="text-embedding-3-small"))
    ask(c, RETURN_ITEM)  # 6 words -> 6 tokens with the fake embedder
    s = c.metrics.summary()
    assert s["embedding_cost_usd"] == pytest.approx(6 * 0.02 / 1_000_000)
    assert s["requests"] == 1  # embedding calls are not user requests


def test_invalid_threshold_is_rejected(cache_path: Path) -> None:
    for threshold in (0, 1.5, -0.1):
        with pytest.raises(ValueError, match="semantic_threshold"):
            OpenAI(
                api_key="sk-test",
                semantic_cache=True,
                semantic_threshold=threshold,
                semantic_embedder=FakeEmbedder(),
                cache_path=cache_path,
            )


def test_invalid_embedder_is_rejected(cache_path: Path) -> None:
    with pytest.raises(TypeError, match="semantic_embedder"):
        OpenAI(
            api_key="sk-test",
            semantic_cache=True,
            semantic_embedder=object(),  # pyright: ignore[reportArgumentType]
            cache_path=cache_path,
        )


# -- default embedder (OpenAI wrapper) ------------------------------------------------


def test_default_embedder_uses_the_wrapped_openai_client(
    server: FakeServer, cache_path: Path
) -> None:
    c = OpenAI(
        api_key="sk-test",
        http_client=server.http_client(),
        semantic_cache=True,
        cache_path=cache_path,
    )
    ask(c, RETURN_ITEM)
    # The fake server returns the same embedding for every text: cosine = 1 -> hit.
    response = ask(c, "Any other wording")
    assert caelum.cache_type(response) == "semantic"
    embedding_calls = [r for r in server.requests if r["path"].endswith("/embeddings")]
    first_call = embedding_calls[0]["body"]
    assert first_call["model"] == "text-embedding-3-small"
    assert first_call["input"] == RETURN_ITEM
    assert c.metrics.summary()["requests"] == 2


def test_openai_embedder_standalone(server: FakeServer) -> None:
    raw = openai.OpenAI(api_key="sk-test", http_client=server.http_client())
    result = OpenAIEmbedder(raw).embed("hello")
    assert result.vector == pytest.approx([0.1, 0.2, 0.3])
    assert result.tokens == 8


async def test_openai_embedder_async(server: FakeServer) -> None:
    raw = openai.AsyncOpenAI(api_key="sk-test", http_client=server.async_http_client())
    result = await OpenAIEmbedder(async_client=raw).aembed("hello")
    assert result.tokens == 8


# -- async ---------------------------------------------------------------------------


async def test_async_semantic_hit_with_default_embedder(
    server: FakeServer, cache_path: Path
) -> None:
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        semantic_cache=True,
        cache_path=cache_path,
    )
    messages: list[Any] = [{"role": "user", "content": RETURN_ITEM}]
    await c.chat.completions.create(model="gpt-4o", messages=messages)
    other: list[Any] = [{"role": "user", "content": SEND_BACK}]
    response = await c.chat.completions.create(model="gpt-4o", messages=other)
    assert caelum.cache_type(response) == "semantic"


async def test_async_with_sync_only_embedder(server: FakeServer, cache_path: Path) -> None:
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        semantic_cache=True,
        semantic_embedder=FakeEmbedder(),
        cache_path=cache_path,
    )
    await c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": RETURN_ITEM}]
    )
    response = await c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": SEND_BACK}]
    )
    assert caelum.cache_type(response) == "semantic"


def test_async_only_embedder_cannot_be_used_synchronously(
    server: FakeServer, cache_path: Path
) -> None:
    c = client(server, cache_path, embedder=AsyncOnlyEmbedder())
    assert content(ask(c, RETURN_ITEM)) == "answer #1"  # falls back to the API


# -- Anthropic -------------------------------------------------------------------------


def test_anthropic_requires_an_explicit_embedder(cache_path: Path) -> None:
    with pytest.raises(ValueError, match="semantic_embedder"):
        Anthropic(api_key="sk-ant-test", semantic_cache=True, cache_path=cache_path)


def test_anthropic_semantic_hit_and_system_isolation(server: FakeServer, cache_path: Path) -> None:
    c = Anthropic(
        api_key="sk-ant-test",
        http_client=server.http_client(),
        semantic_cache=True,
        semantic_embedder=FakeEmbedder(),
        cache_path=cache_path,
    )

    def ask_claude(question: str, system: str) -> Any:
        return c.messages.create(
            model="claude-haiku-5-5",
            max_tokens=64,
            system=system,
            messages=[{"role": "user", "content": question}],
        )

    ask_claude(RETURN_ITEM, "You are a support agent.")
    assert caelum.cache_type(ask_claude(SEND_BACK, "You are a support agent.")) == "semantic"
    assert not caelum.cache_hit(ask_claude(SEND_BACK, "You are a pirate."))


# -- query extraction ------------------------------------------------------------------


def test_query_extraction_rules() -> None:
    base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "Hi there"}]}
    query = openai_semantic_query(base, "ns")
    assert query is not None
    assert query.text == "Hi there"
    same_context = openai_semantic_query(
        {**base, "messages": [{"role": "user", "content": "Hello"}]}, "ns"
    )
    assert same_context is not None
    assert same_context.context_key == query.context_key
    other_ns = openai_semantic_query(base, "other")
    assert other_ns is not None and other_ns.context_key != query.context_key
    assert openai_semantic_query({**base, "stream": True}, "ns") is None
    assert openai_semantic_query({"model": "m", "messages": []}, "ns") is None
    assert openai_semantic_query({"model": "m"}, "ns") is None
    last_is_assistant: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "assistant", "content": "x"}],
    }
    assert openai_semantic_query(last_is_assistant, "ns") is None
    blank: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "   "}]}
    assert openai_semantic_query(blank, "ns") is None
    text_parts: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }
    parts_query = openai_semantic_query(text_parts, "ns")
    assert parts_query is not None and parts_query.text == "Hi"
    huge: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "x" * 30_000}]}
    assert openai_semantic_query(huge, "ns") is None
    anthropic_query = anthropic_semantic_query({**base, "system": "S"}, "ns")
    assert anthropic_query is not None and anthropic_query.context_key != query.context_key


# -- vector store ----------------------------------------------------------------------


def test_vector_store_search_and_isolation() -> None:
    store = SQLiteVectorStore(":memory:")
    store.add("ctx", "model-a", [1, 0, 0], {"answer": 1}, None)
    store.add("ctx", "model-a", [0, 1, 0], {"answer": 2}, None)
    match = store.search("ctx", "model-a", [0.98, 0.2, 0], 0.9)
    assert match is not None and match.value == {"answer": 1}
    assert match.score == pytest.approx(0.98 / math.hypot(0.98, 0.2), abs=1e-6)
    assert store.search("other-ctx", "model-a", [1, 0, 0], 0.5) is None
    assert store.search("ctx", "model-b", [1, 0, 0], 0.5) is None  # other embedding model
    assert store.search("ctx", "model-a", [1, 0], 0.5) is None  # other dimension
    assert len(store) == 2
    store.clear()
    assert len(store) == 0


def test_vector_store_eviction_keeps_newest(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "v.sqlite", max_entries=3)
    for i in range(5):
        store.add("ctx", "m", [1.0, float(i)], {"i": i}, None)
    assert len(store) == 3
    match = store.search("ctx", "m", [1.0, 0.0], 0.0)
    assert match is not None and match.value["i"] >= 2  # entries 0 and 1 were evicted
    store.close()


def test_vector_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "v.sqlite"
    first = SQLiteVectorStore(path)
    first.add("ctx", "m", [0.0, 1.0], {"v": "kept"}, None)
    first.close()
    match = SQLiteVectorStore(path).search("ctx", "m", [0.0, 1.0], 0.99)
    assert match is not None and match.value == {"v": "kept"}


def test_normalize_rejects_degenerate_vectors() -> None:
    assert list(normalize([3, 4])) == pytest.approx([0.6, 0.8])
    for bad in ([], [0, 0], [float("inf"), 1]):
        with pytest.raises(ValueError):
            normalize(bad)
    with pytest.raises(ValueError):
        SQLiteVectorStore(":memory:", max_entries=0)


# -- calibration dataset -----------------------------------------------------------------


def test_calibration_dataset_is_well_formed() -> None:
    data = json.loads((FIXTURES / "semantic_pairs.json").read_text(encoding="utf-8"))
    assert len(data["similar"]) >= 50
    assert len(data["different"]) >= 50
    for pair in [*data["similar"], *data["different"]]:
        assert len(pair) == 2
        assert all(isinstance(text, str) and text.strip() for text in pair)
        assert pair[0] != pair[1]


@pytest.mark.parametrize("use_sumprod", [True, False])
def test_vector_search_without_numpy(monkeypatch: pytest.MonkeyPatch, use_sumprod: bool) -> None:
    """Most users do not install the 'fast' extra: the pure-Python path must agree."""
    from caelum.cache import vectors

    monkeypatch.setattr(vectors, "_np", None)
    if not use_sumprod:
        monkeypatch.setattr(vectors, "_sumprod", None)  # Python 3.10 / 3.11 path
    store = SQLiteVectorStore(":memory:")
    store.add("ctx", "m", [1, 0, 0], {"answer": 1}, None)
    store.add("ctx", "m", [0, 1, 0], {"answer": 2}, None)
    match = store.search("ctx", "m", [0.1, 0.99, 0], 0.9)
    assert match is not None and match.value == {"answer": 2}
    assert store.search("ctx", "m", [0.7, 0.7, 0], 0.9) is None
