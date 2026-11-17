"""Improvement A: net savings = avoided AI calls minus the checks (judge + embeddings)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import OpenAI
from caelum.cache import EmbeddingResult
from caelum.cache.verifier import Verdict
from caelum.journal import Journal, JournalEntry
from tests.conftest import FakeServer

ORIGINAL = "How do I return an item?"
PARAPHRASE = "How can I send back a product?"
VECTORS = {ORIGINAL: [1.0, 0.0], PARAPHRASE: [0.990, 0.14107]}


class Embedder:
    def __init__(self, model: str = "text-embedding-3-small", tokens: int = 1000) -> None:
        self._model, self.tokens = model, tokens

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=VECTORS.get(text, [0.0, 1.0]), tokens=self.tokens)


class Judge:
    def __init__(self, tokens: int = 400, delay: float = 0.0, accept: bool = True) -> None:
        self.model = "gpt-4o-mini"
        self.tokens, self.delay, self.accept = tokens, delay, accept

    def verify(self, stored: str, new: str) -> Verdict:
        time.sleep(self.delay)
        return Verdict(
            accepted=self.accept,
            input_tokens=self.tokens,
            output_tokens=2 if self.tokens else 0,
            calls=2,
        )


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "semantic_cache": True,
        "semantic_embedder": Embedder(),
        "semantic_verifier": Judge(),
        "semantic_threshold": 0.60,
        "cache_path": tmp_path / "net.sqlite",
        "cache_admission": 1,
        "journal_path": tmp_path / "journal.sqlite",
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, question: str, model: str = "gpt-4o") -> Any:
    return c.chat.completions.create(model=model, messages=[{"role": "user", "content": question}])


def journal_of(c: OpenAI) -> Journal:
    assert c.journal is not None
    return c.journal


def test_net_savings_with_real_token_costs(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)  # miss: one embedding
    assert caelum.cache_type(ask(c, PARAPHRASE)) == "semantic"  # hit: embedding + judge x2
    miss, hit = journal_of(c).entries()
    embedding = 1000 * 0.02 / 1_000_000  # text-embedding-3-small
    judge = (400 * 0.15 + 2 * 0.60) / 1_000_000  # gpt-4o-mini
    assert (miss["embedding_cost_usd"], miss["embedding_cost_source"]) == (
        pytest.approx(embedding),
        "tokens",
    )
    assert miss["judge_calls"] == 0 and miss["judge_cost_usd"] == 0
    assert hit["judge_calls"] == 2
    assert (hit["judge_cost_usd"], hit["judge_cost_source"]) == (pytest.approx(judge), "tokens")
    assert hit["overhead_ms"] >= 0
    stats = journal_of(c).stats(None)
    assert stats["verification_cost_usd"] == pytest.approx(2 * embedding + judge)
    assert stats["net_saved_usd"] == pytest.approx(stats["gross_saved_usd"] - 2 * embedding - judge)
    assert stats["gross_saved_usd"] > stats["net_saved_usd"] > 0
    assert stats["alerts"] == []


def test_net_savings_with_estimated_costs(server: FakeServer, tmp_path: Path) -> None:
    c = client(
        server,
        tmp_path,
        semantic_embedder=Embedder(model="my-own-embedder", tokens=0),
        semantic_verifier=Judge(tokens=0),
        judge_call_cost=0.0001,
        embedding_call_cost=0.00001,
        avoided_call_cost=0.002,
    )
    ask(c, ORIGINAL, model="mystery-model")
    ask(c, PARAPHRASE, model="mystery-model")
    hit = journal_of(c).entries()[1]
    assert (hit["embedding_cost_usd"], hit["embedding_cost_source"]) == (0.00001, "estimate")
    assert (hit["judge_cost_usd"], hit["judge_cost_source"]) == (pytest.approx(0.0002), "estimate")
    assert (hit["saved_usd"], hit["saved_source"]) == (0.002, "estimate")
    stats = journal_of(c).stats(None)
    assert stats["net_saved_usd"] == pytest.approx(0.002 - 0.0002 - 2 * 0.00001)


def test_negative_net_savings_raise_a_french_alert(server: FakeServer, tmp_path: Path) -> None:
    c = client(
        server,
        tmp_path,
        semantic_verifier=Judge(tokens=0),
        judge_call_cost=0.01,  # a very expensive judge
        avoided_call_cost=0.0001,  # for very cheap answers
    )
    ask(c, ORIGINAL, model="mystery-model")
    ask(c, PARAPHRASE, model="mystery-model")
    for i in range(8):
        ask(c, f"unique question {i}", model="mystery-model")
    stats = journal_of(c).stats(None)
    assert stats["net_saved_usd"] < 0
    alert = stats["alerts"][0]
    assert alert.startswith("Économie nette négative")
    assert "trop peu de questions répétées" in alert
    assert f"{100 * stats['hit_rate']:.1f} %" in alert


def test_a_cache_slower_than_the_ai_raises_an_alert(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, semantic_verifier=Judge(delay=0.3))
    ask(c, ORIGINAL)
    ask(c, PARAPHRASE)  # served after a 0.3 s judge
    stats = journal_of(c).stats(None)
    assert stats["avg_hit_ms"] > stats["avg_ai_ms"]
    assert stats["avg_check_ms"] >= 300
    assert any(a.startswith("Le cache est plus lent que l'IA directe") for a in stats["alerts"])


def test_break_even_hit_rate(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.sqlite")
    for i in range(10):
        hit = i < 2
        journal.record(
            JournalEntry(
                outcome="hit" if hit else "miss",
                customer="acme",
                saved_usd=0.002 if hit else 0.0,
                saved_source="tokens" if hit else None,
                judge_cost_usd=0.00008,
                embedding_cost_usd=0.00002,
            )
        )
    stats = journal.stats("acme")
    # 0.0001 of checks per request / 0.002 saved per hit = profitable from 5 % of hits.
    assert stats["break_even_hit_rate"] == pytest.approx(0.05)
    assert stats["hit_rate"] == pytest.approx(0.2)
    assert stats["net_saved_usd"] == pytest.approx(0.004 - 0.001)
    assert stats["alerts"] == []
    journal.close()


def test_costs_are_isolated_between_customers(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    with caelum.customer("a"):
        ask(c, ORIGINAL)
        ask(c, PARAPHRASE)
    with caelum.customer("b"):
        ask(c, "Do you sell shoes?")
    a, b = journal_of(c).stats("a"), journal_of(c).stats("b")
    assert a["judge_calls"] == 2 and b["judge_calls"] == 0
    assert b["verification_cost_usd"] == pytest.approx(1000 * 0.02 / 1_000_000)
    assert b["gross_saved_usd"] == 0


def test_a_broken_journal_never_affects_the_answer(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = client(server, tmp_path)

    def broken(rows: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(journal_of(c), "_write", broken)
    ask(c, ORIGINAL)
    assert caelum.cache_type(ask(c, PARAPHRASE)) == "semantic"
    journal_of(c).flush()
    assert c.metrics.summary()["journal_errors"] >= 1
