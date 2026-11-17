"""Task 1: similarity threshold per request, safety floor, strict shortcut, priorities."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import AsyncOpenAI, OpenAI
from caelum.cache import EmbeddingResult
from tests.conftest import FakeServer

ORIGINAL = "How do I return an item?"
PARAPHRASE = "How can I send back a product?"  # 0.990 vs ORIGINAL
CLOSE = "How do I return a gift?"  # 0.950 vs ORIGINAL
VECTORS = {ORIGINAL: [1.0, 0.0], PARAPHRASE: [0.990, 0.14107], CLOSE: [0.950, 0.31225]}


class Embedder:
    @property
    def model(self) -> str:
        return "threshold-embedder"

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=VECTORS.get(text, [0.0, 1.0]), tokens=1)


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "semantic_cache": True,
        "semantic_embedder": Embedder(),
        "semantic_verifier": False,  # defaults: serve from 0.97, gray zone from 0.92
        "cache_path": tmp_path / "t.sqlite",
        "cache_admission": 1,
        "journal_path": tmp_path / "journal.sqlite",
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, question: str, **kwargs: Any) -> Any:
    return c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}], **kwargs
    )


def last_line(c: OpenAI) -> dict[str, Any]:
    journal = c.journal
    assert journal is not None
    return journal.entries()[-1]


# -- 1.1 per-request threshold ----------------------------------------------------------


def test_a_request_can_be_stricter_than_the_client(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    assert not caelum.cache_hit(ask(c, PARAPHRASE, semantic_threshold=0.995))  # 0.990 < 0.995
    line = last_line(c)
    assert (line["threshold"], line["threshold_source"]) == (0.995, "request")
    assert caelum.cache_hit(ask(c, ORIGINAL))  # the client's own setting is unchanged


def test_a_request_can_set_both_bounds(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    response = ask(c, CLOSE, semantic_threshold=0.94, semantic_gray_threshold=0.90)
    assert caelum.cache_type(response) == "semantic"  # 0.950 >= 0.94
    assert "semantic_threshold" not in server.requests[-1]["body"]  # never sent to OpenAI


# -- 1.2 floor, strict shortcut, invalid values ------------------------------------------


def test_a_request_below_the_floor_is_raised_to_the_floor(
    server: FakeServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    c = client(server, tmp_path, semantic_threshold_floor=0.98)
    ask(c, ORIGINAL)
    with caplog.at_level(logging.WARNING, logger="caelum"):
        response = ask(c, CLOSE, semantic_threshold=0.85)  # a mistake in the app
    assert not caelum.cache_hit(response)  # 0.950 < floor 0.98: not served
    line = last_line(c)
    assert (line["threshold"], line["threshold_source"]) == (0.98, "request_floor")
    assert "below the floor" in caplog.text


def test_the_strict_shortcut_asks_for_the_highest_threshold(
    server: FakeServer, tmp_path: Path
) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    assert not caelum.cache_hit(ask(c, PARAPHRASE, semantic_strict=True))  # 0.990 < 0.999
    line = last_line(c)
    assert (line["threshold"], line["threshold_source"]) == (0.999, "strict")
    # The gray zone replaced the stored question by PARAPHRASE: asking it again is a 1.0
    # match, which even the strictest threshold serves.
    assert caelum.cache_hit(ask(c, PARAPHRASE, semantic_strict=True))


@pytest.mark.parametrize(
    "options",
    [
        {"semantic_threshold": 0.5},
        {"semantic_threshold": 1.0},
        {"semantic_threshold": "0.9"},
        {"semantic_threshold": True},
        {"semantic_threshold": 0.95, "semantic_gray_threshold": 0.95},
        {"semantic_threshold": 0.95, "semantic_gray_threshold": 0.99},
        {"semantic_gray_threshold": 0.9},
        {"semantic_strict": "yes"},
        {"semantic_strict": True, "semantic_threshold": 0.99},
    ],
)
def test_invalid_request_thresholds_are_refused_in_french(
    server: FakeServer, tmp_path: Path, options: dict[str, Any]
) -> None:
    c = client(server, tmp_path)
    with pytest.raises(ValueError, match=r"(invalide|doit|demande|fixe)"):
        ask(c, ORIGINAL, **options)
    assert server.calls == 0  # refused before anything is sent


@pytest.mark.parametrize("floor", [0.5, 1.0, "high", True])
def test_an_invalid_floor_is_refused(server: FakeServer, tmp_path: Path, floor: Any) -> None:
    with pytest.raises(ValueError, match="semantic_threshold_floor"):
        client(server, tmp_path, semantic_threshold_floor=floor)
    assert not (tmp_path / "t.sqlite").exists()


# -- 1.3 priority order --------------------------------------------------------------


def test_cache_bypass_wins_over_everything(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    ask(c, ORIGINAL)
    assert not caelum.cache_hit(ask(c, ORIGINAL, cache_bypass=True, semantic_threshold=0.85))
    line = last_line(c)
    assert line["outcome"] == "bypassed" and line["threshold"] is None


def test_exclusions_come_before_the_request_threshold(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    question = "Livrez-vous demain à Bobo ?"
    ask(c, question)
    assert not caelum.cache_hit(ask(c, question, semantic_threshold=0.85))
    line = last_line(c)
    assert line["outcome"] == "excluded" and line["threshold"] is None


def test_the_request_wins_over_the_customer_which_wins_over_the_default(
    server: FakeServer, tmp_path: Path
) -> None:
    c = client(server, tmp_path, semantic_customer_thresholds={"relaxed": (0.90, 0.85)})
    with caelum.customer("relaxed"):
        ask(c, ORIGINAL)
        assert caelum.cache_type(ask(c, CLOSE)) == "semantic"  # customer: 0.950 >= 0.90
        assert last_line(c)["threshold_source"] == "customer"
        ask(c, ORIGINAL)  # restore the original question as the stored one
    with caelum.customer("relaxed"):
        assert not caelum.cache_hit(ask(c, CLOSE, semantic_threshold=0.99))  # request wins
        assert last_line(c)["threshold_source"] == "request"
    with caelum.customer("other"):
        ask(c, ORIGINAL)
        assert not caelum.cache_hit(ask(c, CLOSE))  # default 0.97
        assert (last_line(c)["threshold"], last_line(c)["threshold_source"]) == (0.97, "default")


# -- async -------------------------------------------------------------------------------


def test_the_async_client_supports_per_request_thresholds(
    server: FakeServer, tmp_path: Path
) -> None:
    c = AsyncOpenAI(
        api_key="sk-test",
        http_client=server.async_http_client(),
        semantic_cache=True,
        semantic_embedder=Embedder(),
        semantic_verifier=False,
        cache_path=tmp_path / "a.sqlite",
        cache_admission=1,
        journal_path=tmp_path / "journal-async.sqlite",
        semantic_threshold_floor=0.96,
    )

    async def scenario() -> list[bool]:
        hits: list[bool] = []
        for question, options in (
            (ORIGINAL, {}),
            (CLOSE, {"semantic_threshold": 0.85}),  # raised to the 0.96 floor: 0.950 < 0.96
            (PARAPHRASE, {"semantic_threshold": 0.995}),  # 0.985 vs CLOSE (now stored)
            (PARAPHRASE, {"semantic_strict": True}),  # identical: 1.0
        ):
            create: Any = c.chat.completions.create  # per-request options: untyped
            response = await create(
                model="gpt-4o", messages=[{"role": "user", "content": question}], **options
            )
            hits.append(caelum.cache_hit(response))
        return hits

    assert asyncio.run(scenario()) == [False, False, False, True]
    journal = c.journal
    assert journal is not None
    sources = [r["threshold_source"] for r in journal.entries()]
    assert sources[1:4] == ["request_floor", "request", "strict"]
