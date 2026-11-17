"""Stress test: 10,000 requests through one CAELUM client (cache on, 30% repeats).

Checks that latency stays flat, memory does not grow without bound and the SQLite file
stays small. Fake in-memory OpenAI server (no network, no cost).

    uv run --project packages/python python benchmarks/scripts/bench_stress.py \\
        --output benchmarks/results/$(date +%F)-stress.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import tempfile
import time
import tracemalloc
from pathlib import Path

import httpx2

import caelum
from caelum.cache import SQLiteStore

TOTAL = 10_000
REPEAT_RATIO = 0.30


def handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-stress",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "z" * 500},
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 120, "total_tokens": 180},
        },
    )


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the markdown report to this file")
    args = parser.parse_args()

    rng = random.Random(7)
    asked: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stress.sqlite"
        store = SQLiteStore(path)
        client = caelum.OpenAI(
            api_key="sk-stress",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
            cache=True,
            cache_store=store,
        )
        tracemalloc.start()
        latencies: list[float] = []
        checkpoints: list[tuple[int, float, float]] = []
        started = time.perf_counter()
        for i in range(TOTAL):
            if asked and rng.random() < REPEAT_RATIO:
                question = rng.choice(asked)
            else:
                question = f"Stress question {i}: where is order {i}?"
                asked.append(question)
            t0 = time.perf_counter()
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            if (i + 1) % 2500 == 0:
                current, _ = tracemalloc.get_traced_memory()
                checkpoints.append((i + 1, percentile(latencies[-2500:], 95), current / 1e6))
        duration = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        store.flush()
        size_mb = path.stat().st_size / 1e6
        summary = client.metrics.summary()
        store.close()

    rows = "\n".join(
        f"| {count} | {p95:.3f} | {memory:.1f} |" for count, p95, memory in checkpoints
    )
    spread = " / ".join(f"{percentile(latencies, pct):.3f}" for pct in (50, 95, 99))
    report = f"""# Stress test — {TOTAL:,} requests ({dt.date.today().isoformat()})

One client, exact cache on, {REPEAT_RATIO:.0%} repeated questions, fake in-memory server.

| Metric | Value |
|---|---|
| Duration | {duration:.1f} s ({TOTAL / duration:,.0f} requests/s) |
| Cache hits | {summary["cache_hits"]:,} ({summary["cache_hit_rate"]:.0%}) |
| Latency p50 / p95 / p99 | {spread} ms |
| Peak traced memory | {peak / 1e6:.1f} MB |
| SQLite file size | {size_mb:.1f} MB for {len(asked):,} cached answers |

## Stability (per 2,500 requests)

| Requests | p95 latency (ms) | Traced memory (MB) |
|---|---|---|
{rows}
"""
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
