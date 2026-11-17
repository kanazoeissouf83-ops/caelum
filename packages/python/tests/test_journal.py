"""Task 4: per-request journal and statistics view."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import OpenAI
from caelum.cache import EmbeddingResult
from caelum.cli import main as cli_main
from caelum.journal import Journal, JournalEntry
from tests.conftest import FakeServer

ORIGINAL = "How do I return an item?"
PARAPHRASE = "How can I send back a product?"  # 0.990: served
CLOSE = "How do I return a gift?"  # 0.950: gray zone
OTHER = "Do you sell shoes?"
VECTORS = {
    ORIGINAL: [1.0, 0.0, 0.0],
    PARAPHRASE: [0.990, 0.14107, 0.0],
    CLOSE: [0.950, 0.31225, 0.0],
    OTHER: [0.0, 0.0, 1.0],
}


class Embedder:
    @property
    def model(self) -> str:
        return "journal-embedder"

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=VECTORS.get(text, [0.0, 1.0, 0.0]), tokens=1)


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "cache": True,
        "semantic_cache": True,
        "semantic_embedder": Embedder(),
        "semantic_verifier": False,
        "cache_path": tmp_path / "cache.sqlite",
        "cache_admission": 1,
        "journal_path": tmp_path / "journal.sqlite",
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, question: str, **kwargs: Any) -> Any:
    messages = kwargs.pop("messages", [{"role": "user", "content": question}])
    kwargs.setdefault("model", "gpt-4o")
    return c.chat.completions.create(messages=messages, **kwargs)


def journal_of(c: OpenAI) -> Journal:
    journal = c.journal
    assert journal is not None
    return journal


# -- 4.1 one line per outcome ------------------------------------------------------------------


def test_one_line_per_outcome(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)  # miss
    ask(c, ORIGINAL)  # exact hit
    ask(c, PARAPHRASE)  # semantic hit (0.990)
    ask(c, CLOSE)  # gray zone (0.950)
    ask(c, "Appelez-moi au +226 70 12 34 56")  # excluded
    ask(c, OTHER, cache_bypass=True)  # bypassed
    ask(
        c,
        "",
        messages=[
            {"role": "user", "content": "Livrez-vous à Bobo ?"},
            {"role": "assistant", "content": "Oui."},
            {"role": "user", "content": "et à Ouaga ?"},
        ],
    )  # bypassed (conversation)
    rows = journal_of(c).entries()
    assert [r["outcome"] for r in rows] == [
        "miss", "hit", "hit", "gray", "excluded", "bypassed", "bypassed",
    ]  # fmt: skip
    miss, exact, semantic, gray, excluded, bypassed, conversation = rows
    assert exact["cache_kind"] == "exact" and exact["score"] is None
    assert exact["question"] == exact["origin_question"] == ORIGINAL
    assert semantic["cache_kind"] == "semantic"
    assert semantic["score"] == pytest.approx(0.990, abs=1e-3)
    assert semantic["question"] == PARAPHRASE and semantic["origin_question"] == ORIGINAL
    assert semantic["entry_id"] is not None
    assert gray["score"] == pytest.approx(0.950, abs=1e-3) and gray["saved_usd"] == 0
    assert miss["saved_usd"] == 0 and miss["question"] == ORIGINAL
    assert excluded["reason"] == "phone"
    assert bypassed["reason"] == "cache_bypass" and conversation["reason"] == "conversation"
    for row in rows:
        assert len(row["request_id"]) == 32
        assert row["server"] and row["latency_ms"] >= 0 and row["ts"] > 0
        # Reserved for task 8, never filled yet.
        assert row["shadow_served_answer"] is None and row["shadow_agreement"] is None


def test_hits_record_the_real_cost_or_the_estimate(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, avoided_call_cost=0.01, customer_avoided_call_cost={"vip": 0.05})
    ask(c, ORIGINAL)
    ask(c, ORIGINAL)  # gpt-4o is priced: the real token cost is used
    ask(c, OTHER, model="mystery-model")
    ask(c, OTHER, model="mystery-model")  # unknown price: the estimate is used
    with caelum.customer("vip"):
        ask(c, OTHER, model="mystery-model")
        ask(c, OTHER, model="mystery-model")
    hits = [r for r in journal_of(c).entries() if r["outcome"] == "hit"]
    assert hits[0]["saved_source"] == "tokens" and hits[0]["saved_usd"] > 0
    assert (hits[1]["saved_source"], hits[1]["saved_usd"]) == ("estimate", 0.01)
    assert (hits[2]["saved_source"], hits[2]["saved_usd"]) == ("estimate", 0.05)


# -- 4.2 privacy ----------------------------------------------------------------------------------


def test_no_text_is_logged_for_an_excluded_request(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    secret = "Ma commande n° 4471 n'est pas arrivée"
    ask(c, secret)
    rows = journal_of(c).entries()
    assert rows[0]["outcome"] == "excluded" and rows[0]["reason"] == "order"
    assert rows[0]["question"] is None and rows[0]["origin_question"] is None
    for file in tmp_path.glob("journal.sqlite*"):
        assert b"4471" not in file.read_bytes()


def test_old_lines_are_purged_and_the_size_is_capped(tmp_path: Path) -> None:
    now = [1_000_000.0]
    journal = Journal(tmp_path / "j.sqlite", retention_days=30, max_rows=3, clock=lambda: now[0])
    journal.record(JournalEntry(outcome="miss", ts=now[0] - 31 * 86_400))  # too old
    for i in range(5):
        journal.record(JournalEntry(outcome="miss", ts=now[0] - i))
    journal.purge()
    rows = journal.entries()
    assert len(rows) == 3  # 1 purged by age, 2 dropped by the size cap (oldest first)
    assert [r["ts"] for r in rows] == [now[0] - 2, now[0] - 1, now[0]]
    journal.close()


def test_a_customer_journal_can_be_erased_in_one_call(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    with caelum.customer("a"):
        ask(c, ORIGINAL)
    with caelum.customer("b"):
        ask(c, ORIGINAL)
    journal = journal_of(c)
    assert journal.clear("a") == 1
    assert [r["customer"] for r in journal.entries()] == ["b"]
    assert journal.clear() == 1
    assert journal.entries() == []


def test_the_command_line_erases_the_journal(
    server: FakeServer, tmp_path: Path, capsys: Any
) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    journal_of(c).flush()
    assert cli_main(["journal", "clear", "--path", str(tmp_path / "journal.sqlite")]) == 0
    assert "1 journal line(s) erased" in capsys.readouterr().out
    assert journal_of(c).entries() == []


# -- 4.3 performance and failures -------------------------------------------------------------


def test_a_broken_journal_never_breaks_a_request(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = client(server, tmp_path)
    journal = journal_of(c)

    def broken(rows: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(journal, "_write", broken)
    ask(c, ORIGINAL)
    assert caelum.cache_hit(ask(c, ORIGINAL))  # the requests themselves are fine
    journal.flush()
    assert c.metrics.summary()["journal_errors"] >= 1


def test_writing_the_journal_never_slows_a_request(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = client(server, tmp_path)
    journal = journal_of(c)
    original = journal._write  # pyright: ignore[reportPrivateUsage]

    def slow(rows: Any) -> None:
        time.sleep(1.0)  # a very slow disk
        original(rows)

    monkeypatch.setattr(journal, "_write", slow)
    ask(c, ORIGINAL)
    started = time.perf_counter()
    for _ in range(5):
        ask(c, ORIGINAL)
    assert time.perf_counter() - started < 0.5  # writes happen in the background


def test_an_unopenable_journal_is_skipped(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path, journal_path=tmp_path)  # a directory: cannot be opened
    assert c.journal is None
    ask(c, ORIGINAL)
    assert caelum.cache_hit(ask(c, ORIGINAL))


def test_the_journal_can_be_turned_off(server: FakeServer, tmp_path: Path) -> None:
    assert client(server, tmp_path, journal=False).journal is None
    assert OpenAI(api_key="sk-test", http_client=server.http_client()).journal is None
    with pytest.raises(ValueError):
        client(server, tmp_path, journal_retention_days=0)
    with pytest.raises(ValueError):
        client(server, tmp_path, avoided_call_cost=-1)


# -- 4.5 statistics view ------------------------------------------------------------------------


def test_statistics_only_show_the_customer_own_data(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    with caelum.customer("boutique-a"):
        ask(c, ORIGINAL)
        ask(c, ORIGINAL)
        ask(c, PARAPHRASE)
    with caelum.customer("boutique-b"):
        ask(c, OTHER)
    a = journal_of(c).stats("boutique-a")
    assert a["requests"] == 3 and a["hits"] == 2
    assert a["hit_rate"] == pytest.approx(2 / 3)
    assert a["by_outcome"] == {"hit": 2, "gray": 0, "miss": 1, "excluded": 0, "bypassed": 0}
    assert a["saved_usd"] > 0
    assert all(h["question"] != OTHER for h in a["lowest_score_hits"])
    b = journal_of(c).stats("boutique-b")
    assert b["requests"] == 1 and b["hits"] == 0 and b["lowest_score_hits"] == []


def test_statistics_list_the_twenty_lowest_score_hits(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.sqlite")
    for i in range(25):
        journal.record(
            JournalEntry(
                outcome="hit",
                customer="acme",
                cache_kind="semantic",
                score=0.90 + i / 1000,
                question=f"q{i}",
                origin_question=f"o{i}",
            )
        )
    journal.record(JournalEntry(outcome="hit", customer="acme", cache_kind="exact"))  # no score
    journal.record(JournalEntry(outcome="hit", customer="other", score=0.10, question="x"))
    lowest = journal.stats("acme")["lowest_score_hits"]
    assert len(lowest) == 20
    assert [h["question"] for h in lowest] == [f"q{i}" for i in range(20)]
    assert [h["score"] for h in lowest] == sorted(h["score"] for h in lowest)
    assert lowest[0]["origin_question"] == "o0"
    journal.close()


def test_statistics_over_a_period(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.sqlite")
    now = time.time()
    journal.record(JournalEntry(outcome="hit", ts=now - 3600))  # an hour ago
    journal.record(JournalEntry(outcome="miss", ts=now - 60))  # a minute ago
    assert journal.stats(None, since=now - 600)["requests"] == 1
    assert journal.stats(None, until=now - 600)["by_outcome"]["hit"] == 1
    journal.close()
