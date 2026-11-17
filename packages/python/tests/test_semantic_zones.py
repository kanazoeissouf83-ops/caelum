"""Semantic cache zones: serve (reuse), gray zone (call the API and replace), miss."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletion

import caelum
from caelum import AsyncOpenAI, OpenAI
from caelum.cache import EmbeddingResult
from caelum.cache.semantic import SemanticCache, SemanticZones
from caelum.cache.verifier import Verdict
from tests.conftest import FakeServer

ORIGINAL = "How do I return an item?"
PARAPHRASE = "How can I send back a product?"  # cosine 0.990 -> serve (default 0.97)
CLOSE = "How do I return a gift?"  # cosine 0.950 -> gray zone (default 0.92-0.97)
RELATED = "What is the return window for shoes?"  # cosine 0.900 -> miss (default)

VECTORS: dict[str, list[float]] = {
    ORIGINAL: [1.0, 0.0],
    PARAPHRASE: [0.990, 0.14107],
    CLOSE: [0.950, 0.31225],
    RELATED: [0.900, 0.43589],
}


class Embedder:
    @property
    def model(self) -> str:
        return "zones-embedder"

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=VECTORS[text], tokens=1)


class RejectingVerifier:
    """A judge that refuses every candidate."""

    model = "fake-judge"

    def verify(self, stored: str, new: str) -> Verdict:
        return Verdict(accepted=False)


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "semantic_cache": True,
        "semantic_embedder": Embedder(),
        "semantic_verifier": False,
        "cache_path": tmp_path / "zones.sqlite",
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: OpenAI, question: str) -> ChatCompletion:
    return c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )


def semantic(c: Any) -> SemanticCache:
    cache = c.__dict__["_caelum_state"].semantic_cache
    assert cache is not None
    return cast(SemanticCache, cache)


def stored_answers(c: Any) -> int:
    return len(semantic(c).store)


# -- the three zones ----------------------------------------------------------------


def test_high_score_is_served_from_the_cache(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    response = ask(c, PARAPHRASE)
    assert caelum.cache_type(response) == "semantic"
    assert server.calls == 1
    assert c.metrics.summary()["semantic_refreshes"] == 0


def test_gray_zone_calls_the_api_and_replaces_the_entry(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    response = ask(c, CLOSE)
    assert not caelum.cache_hit(response)
    assert server.calls == 2
    assert response.choices[0].message.content == "answer #2"
    # The close-but-uncertain entry was replaced, not duplicated.
    assert stored_answers(c) == 1
    assert c.metrics.summary()["semantic_refreshes"] == 1
    # The fresh answer is the one now stored: the same question is served from it.
    again = ask(c, CLOSE)
    assert caelum.cache_type(again) == "semantic"
    assert again.choices[0].message.content == "answer #2"
    assert server.calls == 2


def test_low_score_calls_the_api_and_keeps_both_entries(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    response = ask(c, RELATED)
    assert not caelum.cache_hit(response)
    assert server.calls == 2
    assert stored_answers(c) == 2
    assert c.metrics.summary()["semantic_refreshes"] == 0


def test_both_bounds_are_adjustable(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, semantic_threshold=0.995, semantic_gray_threshold=0.85)
    ask(c, ORIGINAL)
    assert not caelum.cache_hit(ask(c, PARAPHRASE))  # 0.990: now in the gray zone
    assert stored_answers(c) == 1
    assert not caelum.cache_hit(ask(c, ORIGINAL))  # 0.990 vs the stored paraphrase: gray
    assert not caelum.cache_hit(ask(c, RELATED))  # 0.900 >= 0.85: gray as well
    assert stored_answers(c) == 1
    assert c.metrics.summary()["semantic_refreshes"] == 3
    assert server.calls == 4


def test_gray_threshold_equal_to_threshold_disables_the_gray_zone(
    server: FakeServer, tmp_path: Path
) -> None:
    c = client(server, tmp_path, semantic_threshold=0.97, semantic_gray_threshold=0.97)
    ask(c, ORIGINAL)
    ask(c, CLOSE)
    assert stored_answers(c) == 2  # plain miss, nothing replaced
    assert c.metrics.summary()["semantic_refreshes"] == 0


def test_default_gray_zone_is_just_below_the_threshold(server: FakeServer, tmp_path: Path) -> None:
    zones = semantic(client(server, tmp_path)).default_zones
    assert zones == SemanticZones(serve=0.97, refresh=0.92)
    custom = semantic(client(server, tmp_path, semantic_threshold=0.9)).default_zones
    assert custom.refresh == pytest.approx(0.85)


# -- per customer -------------------------------------------------------------------


def test_bounds_can_be_set_per_customer(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, semantic_customer_thresholds={"strict": (0.995, 0.98)})
    with caelum.customer("strict"):
        ask(c, ORIGINAL)
        assert not caelum.cache_hit(ask(c, PARAPHRASE))  # 0.990 is gray for "strict"
    assert c.metrics.summary()["semantic_refreshes"] == 1
    # Another customer has its own cache and keeps the default bounds: 0.990 is served.
    with caelum.customer("relaxed"):
        ask(c, ORIGINAL)
        assert caelum.cache_type(ask(c, PARAPHRASE)) == "semantic"


def test_bounds_can_change_at_runtime(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    cache = semantic(c)
    cache.set_thresholds("acme", 0.85, 0.80)
    assert cache.zones_for("acme") == SemanticZones(serve=0.85, refresh=0.80)
    with caelum.customer("acme"):
        ask(c, ORIGINAL)
        assert caelum.cache_type(ask(c, RELATED)) == "semantic"  # 0.900 >= 0.85
    cache.reset_thresholds("acme")
    assert cache.zones_for("acme") == cache.default_zones


# -- interplay and safety -----------------------------------------------------------


def test_a_rejected_candidate_is_a_plain_miss(server: FakeServer, tmp_path: Path) -> None:
    # The judge refused the match: the stored answer belongs to another question, so it
    # is kept (not replaced) and the new answer is added.
    c = client(server, tmp_path, semantic_verifier=RejectingVerifier(), semantic_threshold=0.98)
    ask(c, ORIGINAL)
    assert not caelum.cache_hit(ask(c, PARAPHRASE))
    assert stored_answers(c) == 2


def test_gray_zone_works_with_the_async_client(server: FakeServer, tmp_path: Path) -> None:
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        semantic_cache=True,
        semantic_embedder=Embedder(),
        semantic_verifier=False,
        cache_path=tmp_path / "zones-async.sqlite",
    )

    async def scenario() -> None:
        await c.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": ORIGINAL}]
        )
        response = await c.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": CLOSE}]
        )
        assert not caelum.cache_hit(response)

    asyncio.run(scenario())
    assert server.calls == 2
    assert stored_answers(c) == 1
    assert c.metrics.summary()["semantic_refreshes"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"semantic_threshold": 0.9, "semantic_gray_threshold": 0.95},
        {"semantic_gray_threshold": 0},
        {"semantic_gray_threshold": "0.9"},
        {"semantic_customer_thresholds": {"acme": (0.9,)}},
        {"semantic_customer_thresholds": {"acme": (0.9, 0.95)}},
        {"semantic_customer_thresholds": {"": (0.9, 0.85)}},
        {"semantic_customer_thresholds": [("acme", (0.9, 0.85))]},
    ],
)
def test_wrong_bounds_are_refused_at_construction(
    server: FakeServer, tmp_path: Path, options: dict[str, Any]
) -> None:
    with pytest.raises((ValueError, TypeError)):
        client(server, tmp_path, **options)
