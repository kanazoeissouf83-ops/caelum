"""LLM verification of semantic-cache matches."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import openai
import pytest

import caelum
from caelum import Anthropic, AsyncOpenAI, LLMVerifier, OpenAI
from caelum.cache.vectors import SQLiteVectorStore
from caelum.cache.verifier import Verdict, parse_verdict
from tests.conftest import FakeServer
from tests.test_cache_semantic import RETURN_ITEM, SEND_BACK, SHOES_WINDOW, FakeEmbedder, ask

# Fake verdict usage: 150 input + 1 output tokens on gpt-4o-mini ($0.15 / $0.60 per 1M).
VERDICT_COST = (150 * 0.15 + 1 * 0.60) / 1_000_000


def verified_client(server: FakeServer, cache_path: Path, **kwargs: Any) -> OpenAI:
    """Semantic cache with the default verifier (gpt-4o-mini through the same client)."""
    options: dict[str, Any] = {
        "semantic_cache": True,
        "semantic_embedder": FakeEmbedder(),
        "cache_path": cache_path,
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


class FakeVerifier:
    """Records calls and answers with a fixed verdict."""

    def __init__(self, accept: bool = True, error: Exception | None = None) -> None:
        self.accept = accept
        self.error = error
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "gpt-4o-mini"

    def verify(self, stored: str, new: str) -> Verdict:
        self.calls.append((stored, new))
        if self.error is not None:
            raise self.error
        return Verdict(accepted=self.accept, input_tokens=10, output_tokens=1)


class AsyncOnlyVerifier:
    @property
    def model(self) -> str:
        return "gpt-4o-mini"

    async def averify(self, stored: str, new: str) -> Verdict:
        return Verdict(accepted=True)


# -- verdict parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("YES", True), ("yes", True), ("Yes.", True), ("**YES**", True), ("NO", False),
     ("No.", False), ("", False), ("Yesterday", False), ("I think YES", False)],
)  # fmt: skip
def test_parse_verdict_only_accepts_a_clear_yes(text: str, expected: bool) -> None:
    assert parse_verdict(text) is expected


# -- LLMVerifier ----------------------------------------------------------------------


def test_llm_verifier_asks_both_orders(server: FakeServer) -> None:
    verifier = LLMVerifier(openai.OpenAI(api_key="sk-test", http_client=server.http_client()))
    verdict = verifier.verify(RETURN_ITEM, SEND_BACK)
    assert verdict.accepted
    assert server.verifier_calls == 2
    assert verdict.input_tokens == 300
    questions = [r["body"]["messages"][1]["content"] for r in server.requests]
    assert questions[0].startswith(f'Question 1: "{RETURN_ITEM}"')
    assert questions[1].startswith(f'Question 1: "{SEND_BACK}"')
    assert server.requests[0]["body"]["temperature"] == 0


def test_llm_verifier_stops_at_the_first_no(server: FakeServer) -> None:
    server.verifier_answer = "NO"
    verifier = LLMVerifier(openai.OpenAI(api_key="sk-test", http_client=server.http_client()))
    assert not verifier.verify(RETURN_ITEM, SHOES_WINDOW).accepted
    assert server.verifier_calls == 1


def test_llm_verifier_single_order(server: FakeServer) -> None:
    raw = openai.OpenAI(api_key="sk-test", http_client=server.http_client())
    assert LLMVerifier(raw, symmetric=False).verify(RETURN_ITEM, SEND_BACK).accepted
    assert server.verifier_calls == 1


async def test_llm_verifier_async_runs_both_orders(server: FakeServer) -> None:
    raw = openai.AsyncOpenAI(api_key="sk-test", http_client=server.async_http_client())
    verdict = await LLMVerifier(async_client=raw).averify(RETURN_ITEM, SEND_BACK)
    assert verdict.accepted
    assert server.verifier_calls == 2
    single = await LLMVerifier(async_client=raw, symmetric=False).averify(RETURN_ITEM, SEND_BACK)
    assert single.accepted
    assert server.verifier_calls == 3


# -- cache behaviour ------------------------------------------------------------------


def test_default_verifier_accepts_a_paraphrase(server: FakeServer, cache_path: Path) -> None:
    c = verified_client(server, cache_path)
    first = ask(c, RETURN_ITEM)
    second = ask(c, SEND_BACK)
    assert server.calls == 1
    assert caelum.cache_type(second) == "semantic"
    assert second.choices[0].message.content == first.choices[0].message.content
    summary = c.metrics.summary()
    assert summary["semantic_verifications"] == 1
    assert summary["semantic_rejections"] == 0
    assert summary["verifier_cost_usd"] == pytest.approx(2 * VERDICT_COST)


def test_verifier_rejection_calls_the_api(server: FakeServer, cache_path: Path) -> None:
    server.verifier_answer = "NO"
    c = verified_client(server, cache_path)
    ask(c, RETURN_ITEM)
    second = ask(c, SHOES_WINDOW)  # cosine ~0.90: a candidate, but a different question
    assert server.calls == 2
    assert not caelum.cache_hit(second)
    assert c.metrics.summary()["semantic_rejections"] == 1


def test_verifier_lowers_the_default_threshold(server: FakeServer, cache_path: Path) -> None:
    c = verified_client(server, cache_path)
    ask(c, RETURN_ITEM)
    ask(c, SHOES_WINDOW)  # 0.90 >= 0.60: the judge (here: YES) decides
    assert server.calls == 1
    strict = verified_client(server, cache_path.with_name("strict.db"), semantic_verifier=False)
    ask(strict, RETURN_ITEM)
    ask(strict, SHOES_WINDOW)  # 0.90 < 0.97 without a judge
    assert server.calls == 3
    assert strict.metrics.summary()["semantic_verifications"] == 0


def test_explicit_threshold_is_kept_with_a_verifier(server: FakeServer, cache_path: Path) -> None:
    verifier = FakeVerifier()
    c = verified_client(server, cache_path, semantic_verifier=verifier, semantic_threshold=0.95)
    ask(c, RETURN_ITEM)
    ask(c, SHOES_WINDOW)
    assert verifier.calls == []  # below 0.95: never sent to the judge
    assert server.calls == 2


def test_custom_verifier_receives_stored_and_new_question(
    server: FakeServer, cache_path: Path
) -> None:
    verifier = FakeVerifier()
    c = verified_client(server, cache_path, semantic_verifier=verifier)
    ask(c, RETURN_ITEM)
    ask(c, SEND_BACK)
    assert verifier.calls == [(RETURN_ITEM, SEND_BACK)]
    assert server.verifier_calls == 0


def test_verifier_failure_falls_back_to_the_api(server: FakeServer, cache_path: Path) -> None:
    verifier = FakeVerifier(error=RuntimeError("judge down"))
    c = verified_client(server, cache_path, semantic_verifier=verifier)
    ask(c, RETURN_ITEM)
    second = ask(c, SEND_BACK)
    assert server.calls == 2
    assert not caelum.cache_hit(second)


def test_async_only_verifier_is_skipped_by_sync_clients(
    server: FakeServer, cache_path: Path
) -> None:
    c = verified_client(server, cache_path, semantic_verifier=AsyncOnlyVerifier())
    ask(c, RETURN_ITEM)
    ask(c, SEND_BACK)
    assert server.calls == 2


async def test_async_client_uses_the_default_verifier(server: FakeServer, cache_path: Path) -> None:
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        semantic_cache=True,
        semantic_embedder=FakeEmbedder(),
        cache_path=cache_path,
    )
    messages: list[Any] = [{"role": "user", "content": RETURN_ITEM}]
    await c.chat.completions.create(model="gpt-4o", messages=messages)
    again = await c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": SEND_BACK}]
    )
    assert caelum.cache_type(again) == "semantic"
    assert server.verifier_calls == 2


async def test_async_client_with_sync_verifier(server: FakeServer, cache_path: Path) -> None:
    verifier = FakeVerifier()
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        semantic_cache=True,
        semantic_embedder=FakeEmbedder(),
        semantic_verifier=verifier,
        cache_path=cache_path,
    )
    await c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": RETURN_ITEM}]
    )
    again = await c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": SEND_BACK}]
    )
    assert caelum.cache_hit(again)
    assert verifier.calls == [(RETURN_ITEM, SEND_BACK)]


def test_entries_without_question_text_are_never_verified_hits(
    server: FakeServer, cache_path: Path
) -> None:
    c = verified_client(server, cache_path)
    ask(c, RETURN_ITEM)
    with closing(sqlite3.connect(cache_path)) as conn, conn:  # a cache file from before 0.1
        conn.execute("UPDATE caelum_semantic SET query = NULL")
    c2 = verified_client(server, cache_path)
    ask(c2, SEND_BACK)
    assert server.calls == 2
    assert server.verifier_calls == 0


def test_old_cache_files_gain_the_question_column(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE caelum_semantic (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " context_key TEXT NOT NULL, embedder TEXT NOT NULL, vector BLOB NOT NULL,"
            " value TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL)"
        )
    store = SQLiteVectorStore(path)
    store.add("ctx", "emb", [1.0, 0.0], {"a": 1}, None, query="hello")
    match = store.search("ctx", "emb", [1.0, 0.0], 0.5)
    assert match is not None and match.query == "hello"
    store.close()


def test_anthropic_has_no_default_verifier(cache_path: Path) -> None:
    with pytest.raises(ValueError, match="semantic_verifier=True"):
        Anthropic(
            api_key="sk-ant-test",
            semantic_cache=True,
            semantic_embedder=FakeEmbedder(),
            semantic_verifier=True,
            cache_path=cache_path,
        )


def test_invalid_verifier_is_rejected(cache_path: Path) -> None:
    with pytest.raises(TypeError, match="semantic_verifier"):
        OpenAI(
            api_key="sk-test",
            semantic_cache=True,
            semantic_embedder=FakeEmbedder(),
            semantic_verifier=object(),  # type: ignore[arg-type]
            cache_path=cache_path,
        )
