"""Metrics and pricing: the developer sees what CAELUM saves, in one call."""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import openai
import pytest

from caelum import AsyncOpenAI, Metrics, OpenAI, pricing
from caelum.utils import extract_usage
from tests.conftest import FakeServer

MESSAGES: list[Any] = [{"role": "user", "content": "How do I return an item?"}]
# Fake server usage: 1,000 input + 500 output tokens on gpt-4o ($2.50 / $10 per 1M).
GPT4O_CALL_COST = (1000 * 2.50 + 500 * 10.00) / 1_000_000


def client(server: FakeServer, **kwargs: Any) -> OpenAI:
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **kwargs)


def ask(c: OpenAI, **kwargs: Any) -> Any:
    params: dict[str, Any] = {"model": "gpt-4o", "messages": MESSAGES, **kwargs}
    return cast(Any, c.chat.completions.create(**params))


# -- integration with the client ------------------------------------------------


def test_metrics_with_cache(server: FakeServer, cache_path: Path) -> None:
    c = client(server, cache=True, cache_path=cache_path)
    ask(c)
    ask(c)
    s = c.metrics.summary()
    assert s["requests"] == 2
    assert s["cache_hits"] == s["cache_hits_exact"] == 1
    assert s["cache_misses"] == 1
    assert s["api_calls"] == 1
    assert s["cache_hit_rate"] == 0.5
    assert s["cost_usd"] == pytest.approx(GPT4O_CALL_COST)
    assert s["cost_saved_usd"] == pytest.approx(GPT4O_CALL_COST)
    assert s["requests_by_model"] == {"gpt-4o": 2}
    assert s["latency_avg_ms"] > 0


def test_metrics_without_cache(server: FakeServer) -> None:
    c = client(server)
    ask(c)
    s = c.metrics.summary()
    assert (s["requests"], s["cache_hits"], s["cache_misses"]) == (1, 0, 0)
    assert s["cost_usd"] == pytest.approx(GPT4O_CALL_COST)


def test_failures_are_counted_and_reraised(server: FakeServer) -> None:
    server.failures = [400]
    c = client(server)
    with pytest.raises(openai.BadRequestError):
        ask(c)
    s = c.metrics.summary()
    assert s["requests"] == 1
    assert s["failed_requests"] == 1
    assert s["api_calls"] == 0


def test_streams_are_counted_without_cost(server: FakeServer) -> None:
    c = client(server)
    list(ask(c, stream=True))
    s = c.metrics.summary()
    assert s["requests"] == 1
    assert s["unpriced_requests"] == 1


def test_unknown_model_is_unpriced(server: FakeServer) -> None:
    c = client(server)
    ask(c, model="my-private-model")
    assert c.metrics.summary()["unpriced_requests"] == 1
    assert c.metrics.summary()["cost_usd"] == 0


def test_embeddings_cost(server: FakeServer) -> None:
    c = client(server)
    c.embeddings.create(model="text-embedding-3-small", input="hi")
    assert c.metrics.summary()["cost_usd"] == pytest.approx(8 * 0.02 / 1_000_000)


def test_copies_share_metrics(server: FakeServer) -> None:
    c = client(server)
    ask(c.with_options(timeout=5))
    assert c.metrics.summary()["requests"] == 1


async def test_async_client_metrics(server: FakeServer) -> None:
    c = AsyncOpenAI(api_key="sk-test", http_client=server.async_http_client())
    await c.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert c.metrics.summary()["requests"] == 1


# -- Metrics object ---------------------------------------------------------------


def test_print_summary_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rich.console", None)  # simulate rich not installed
    m = Metrics()
    m.record_api_call(model="gpt-4o", latency_ms=200, cost_usd=1.0, cache_miss=True)
    m.record_cache_hit(kind="exact", model="gpt-4o", latency_ms=1, saved_usd=24.3)
    out = io.StringIO()
    m.print_summary(file=out)
    text = out.getvalue()
    assert text.startswith("CAELUM Local Metrics")
    assert "Cache hits:" in text
    assert "(50%)" in text
    assert "$ 24.30" in text


def test_print_summary_with_rich() -> None:
    m = Metrics()
    m.record_retry()
    m.record_failure(model=None, latency_ms=3)
    m.record_api_call(model="x", latency_ms=1, cost_usd=None, cache_miss=False)
    out = io.StringIO()
    m.print_summary(file=out)
    assert "CAELUM Local Metrics" in out.getvalue()
    assert "Retries" in out.getvalue()
    assert "cost unknown" in m.format_summary()


def test_thousands_separator() -> None:
    m = Metrics()
    for _ in range(1240):
        m.record_api_call(model="m", latency_ms=1, cost_usd=0, cache_miss=False)
    assert "1 240" in m.format_summary()


def test_percentiles_and_bounded_samples() -> None:
    m = Metrics(max_latency_samples=100)
    for i in range(1, 201):
        m.record_api_call(model="m", latency_ms=float(i), cost_usd=0.0, cache_miss=False)
    s = m.summary()
    assert len(m.latencies) == 100  # only the most recent samples are kept
    assert s["latency_p50_ms"] == pytest.approx(150, abs=1)
    assert s["latency_p95_ms"] == pytest.approx(195, abs=1)
    assert s["requests"] == 200


def test_reset_and_export_json(tmp_path: Path) -> None:
    m = Metrics()
    m.record_fallback()
    m.record_embedding_cost(0.5)
    m.record_embedding_cost(None)
    m.record_cache_hit(kind="semantic", model="m", latency_ms=1, saved_usd=None)
    path = tmp_path / "metrics.json"
    m.export_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fallbacks"] == 1
    assert data["cache_hits_semantic"] == 1
    assert data["embedding_cost_usd"] == 0.5
    assert "requests=1" in repr(m)
    m.reset()
    assert m.summary()["requests"] == 0
    assert m.summary()["latency_p95_ms"] == 0


def test_metrics_are_thread_safe() -> None:
    m = Metrics()

    def work() -> None:
        for _ in range(1000):
            m.record_api_call(model="m", latency_ms=1, cost_usd=0.001, cache_miss=True)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.summary()["requests"] == 8000
    assert m.summary()["cost_usd"] == pytest.approx(8.0)


# -- pricing ------------------------------------------------------------------------


def test_dated_snapshots_resolve_to_the_longest_prefix() -> None:
    assert pricing.get_price("gpt-4o-2024-08-06") == pricing.PRICING["gpt-4o"]
    assert pricing.get_price("gpt-4o-mini-2024-07-18") == pricing.PRICING["gpt-4o-mini"]
    assert pricing.get_price("gpt-4o") == pricing.PRICING["gpt-4o"]
    assert pricing.get_price("gpt-4oops") is None
    assert pricing.get_price("claude-opus-5-5") == {"input_per_1m": 4.0, "output_per_1m": 20.0}


def test_estimate_cost() -> None:
    assert pricing.estimate_cost("gpt-4o", 1000, 500) == pytest.approx(GPT4O_CALL_COST)
    assert pricing.estimate_cost("unknown-model", 1, 1) is None


def test_register_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pricing, "PRICING", dict(pricing.PRICING))
    pricing.register_price("my-model", 1.0, 2.0)
    assert pricing.estimate_cost("my-model", 1_000_000, 1_000_000) == 3.0
    with pytest.raises(ValueError):
        pricing.register_price("bad", -1, 0)


def test_extract_usage_supports_both_providers() -> None:
    openai_like = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    anthropic_like = SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=3))
    embedding_like = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=4, total_tokens=4))
    assert extract_usage(openai_like) == (10, 5)
    assert extract_usage(anthropic_like) == (7, 3)
    assert extract_usage(embedding_like) == (4, 0)
    assert extract_usage(SimpleNamespace()) is None
    assert extract_usage(SimpleNamespace(usage=SimpleNamespace())) is None
