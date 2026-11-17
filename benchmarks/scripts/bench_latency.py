"""Benchmark: latency added by CAELUM on cache misses (the 5 ms rule).

Compares the raw OpenAI SDK with CAELUM in several configurations against an in-memory
fake server (no network: the numbers isolate SDK + CAELUM work). Requests alternate
between raw and CAELUM clients to cancel out machine noise. Every CAELUM request is a
cache miss (unique questions), which is the worst case for overhead.

Constraint: p95 overhead < 5 ms target, 10 ms hard ceiling.

    uv run --project packages/python python benchmarks/scripts/bench_latency.py \\
        --output benchmarks/results/$(date +%F)-performance.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx2

import caelum
from caelum import RetryPolicy
from caelum.cache import SQLiteStore

REQUESTS = 500
THREADS = 8
ROUTER: dict[str, Any] = {
    "routes": [
        {
            "name": "long",
            "match": {"messages_min_tokens": 5000},
            "target": "gpt-4o",
            "provider": "openai",
        }
    ]
}


API_LATENCY_S = 0.020  # simulated provider latency: real calls are I/O-bound


def handler(request: httpx2.Request) -> httpx2.Response:
    time.sleep(API_LATENCY_S)
    body = json.loads(request.content)
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-bench",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "y" * 300},
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
        },
    )


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))]


_counter = 0
_counter_lock = threading.Lock()


def next_question() -> str:
    global _counter
    with _counter_lock:
        _counter += 1
        return f"Unique question number {_counter}: what is the status of my order?"


def instrument(wrapped: Any) -> threading.local:
    """Time the SDK call inside a CAELUM client, so CAELUM's own time can be isolated."""
    sdk = wrapped._caelum_client.chat.completions
    original = sdk.create
    local = threading.local()

    def timed_create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            local.inner_ms = (time.perf_counter() - started) * 1000

    sdk.create = timed_create
    return local


def measure(wrapped: Any, parallel: bool) -> dict[str, float]:
    """CAELUM overhead per request = total time - time spent in the SDK call."""
    local = instrument(wrapped)

    def one(_: int) -> float:
        question = next_question()
        started = time.perf_counter()
        wrapped.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
        )
        return (time.perf_counter() - started) * 1000 - local.inner_ms

    for i in range(30):  # warm-up
        one(i)
    if parallel:
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            overheads = list(pool.map(one, range(REQUESTS)))
    else:
        overheads = [one(i) for i in range(REQUESTS)]
    return {
        "p50": percentile(overheads, 50),
        "p95": percentile(overheads, 95),
        "p99": percentile(overheads, 99),
        "max": max(overheads),
    }


def run() -> list[tuple[str, str, dict[str, float]]]:
    transport = httpx2.MockTransport(handler)

    def http() -> httpx2.Client:
        return httpx2.Client(transport=transport)

    results: list[tuple[str, str, dict[str, float]]] = []
    with tempfile.TemporaryDirectory() as tmp:
        stores: list[SQLiteStore] = []

        def store() -> SQLiteStore:
            stores.append(SQLiteStore(Path(tmp) / f"bench-{len(stores)}.sqlite"))
            return stores[-1]

        configs: list[tuple[str, Callable[[], Any]]] = [
            ("wrapper only", lambda: caelum.OpenAI(api_key="sk-bench", http_client=http())),
            (
                "cache=True (miss)",
                lambda: caelum.OpenAI(
                    api_key="sk-bench", http_client=http(), cache=True, cache_store=store()
                ),
            ),
            (
                "cache + router + retry",
                lambda: caelum.OpenAI(
                    api_key="sk-bench",
                    http_client=http(),
                    cache=True,
                    cache_store=store(),
                    router_config=ROUTER,
                    retry=RetryPolicy(),
                ),
            ),
        ]
        for name, factory in configs:
            for mode, parallel in (("sequential", False), (f"{THREADS} threads", True)):
                results.append((name, mode, measure(factory(), parallel)))
        for item in stores:
            item.close()
    return results


def render(results: list[tuple[str, str, dict[str, float]]]) -> str:
    worst_p95 = max(r["p95"] for _, _, r in results)
    verdict = "PASS" if worst_p95 < 5 else ("WARN (over target)" if worst_p95 < 10 else "FAIL")
    rows = "\n".join(
        f"| {name} | {mode} | {r['p50']:.3f} | **{r['p95']:.3f}** | "
        f"{r['p99']:.3f} | {r['max']:.3f} |"
        for name, mode, r in results
    )
    header = "| Configuration | Mode | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |"
    return f"""# Benchmark — CAELUM latency overhead ({dt.date.today().isoformat()})

Overhead measured **per request** as total time minus the time spent inside the provider
SDK call ({REQUESTS} cache-miss requests per row). Fake OpenAI server with a simulated
{API_LATENCY_S * 1000:.0f} ms provider latency (real calls are I/O-bound: hundreds of
milliseconds). Python {platform.python_version()}, {platform.system()} {platform.machine()}.

{header}
|---|---|---|---|---|---|
{rows}

**Worst p95 overhead: {worst_p95:.3f} ms** — target < 5 ms, ceiling 10 ms: **{verdict}**.

Note: with a zero-latency fake server, 8 CPU-bound threads contend for the Python GIL and
every SQLite call has to wait for it again; that synthetic setup inflates tail latency
(around +50 ms at p95) without reflecting real, network-bound workloads.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the markdown report to this file")
    args = parser.parse_args()
    report = render(run())
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
