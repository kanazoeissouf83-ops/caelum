"""Benchmark: exact cache savings and CAELUM overhead.

Simulates 1,000 chat requests where 30% are duplicates of earlier requests, against an
in-memory fake OpenAI server (no network, no cost). Measures:

* API calls (and therefore cost) saved by the exact cache;
* latency added by CAELUM on cache misses, compared with the raw OpenAI SDK
  (constraint: p95 < 5 ms target, 10 ms hard ceiling);
* latency of cache hits.

Run from the repository root:

    uv run --project packages/python python benchmarks/scripts/bench_cache_exact.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx2
import openai

import caelum
from caelum.cache import SQLiteStore

TOTAL_REQUESTS = 1_000
DUPLICATE_RATIO = 0.30
SEED = 2026


def fake_openai_handler(request: httpx2.Request) -> httpx2.Response:
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
                    "message": {"role": "assistant", "content": "x" * 400},
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 100, "total_tokens": 220},
        },
    )


def build_workload() -> list[str]:
    """Return the request sequence: unique questions plus repeats of earlier ones."""
    rng = random.Random(SEED)
    unique_count = round(TOTAL_REQUESTS * (1 - DUPLICATE_RATIO))
    workload = [f"Customer question #{i}: how do I track order {i}?" for i in range(unique_count)]
    rng.shuffle(workload)
    for _ in range(TOTAL_REQUESTS - unique_count):
        position = rng.randint(1, len(workload))
        # A duplicate always repeats a question asked *before* it.
        workload.insert(position, rng.choice(workload[:position]))
    return workload


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def timed_calls(client: Any, questions: list[str]) -> tuple[list[float], list[bool]]:
    durations: list[float] = []
    hits: list[bool] = []
    for question in questions:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
        )
        durations.append((time.perf_counter() - started) * 1000)
        hits.append(caelum.cache_hit(response))
    return durations, hits


def run() -> dict[str, Any]:
    questions = build_workload()
    transport = httpx2.MockTransport(fake_openai_handler)

    raw = openai.OpenAI(api_key="sk-bench", http_client=httpx2.Client(transport=transport))
    timed_calls(raw, questions[:50])  # warm-up
    raw_ms, _ = timed_calls(raw, questions)

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(Path(tmp) / "bench.sqlite")
        wrapped = caelum.OpenAI(
            api_key="sk-bench",
            http_client=httpx2.Client(transport=transport),
            cache=True,
            cache_store=store,
        )
        cached_ms, hits = timed_calls(wrapped, questions)
        store.close()  # release the file before the temporary directory is removed

    miss_ms = [d for d, hit in zip(cached_ms, hits, strict=True) if not hit]
    hit_ms = [d for d, hit in zip(cached_ms, hits, strict=True) if hit]
    api_calls = len(miss_ms)
    return {
        "requests": len(questions),
        "unique_questions": len(set(questions)),
        "api_calls": api_calls,
        "api_calls_saved": len(hit_ms),
        "savings_pct": 100 * len(hit_ms) / len(questions),
        "raw_p50": statistics.median(raw_ms),
        "raw_p95": percentile(raw_ms, 95),
        "miss_p50": statistics.median(miss_ms),
        "miss_p95": percentile(miss_ms, 95),
        "hit_p50": statistics.median(hit_ms),
        "hit_p95": percentile(hit_ms, 95),
    }


def render(results: dict[str, Any]) -> str:
    overhead_p50 = results["miss_p50"] - results["raw_p50"]
    overhead_p95 = results["miss_p95"] - results["raw_p95"]
    verdict = "PASS" if overhead_p95 < 5 else ("WARN" if overhead_p95 < 10 else "FAIL")
    today = dt.date.today().isoformat()
    return f"""# Benchmark — exact cache ({today})

Workload: {results["requests"]} chat requests, {DUPLICATE_RATIO:.0%} duplicates
({results["unique_questions"]} unique questions), fake in-memory OpenAI server
(no network: latencies below measure SDK + CAELUM work only).

## Savings

| Metric | Value |
|---|---|
| Requests | {results["requests"]} |
| API calls made | {results["api_calls"]} |
| API calls saved (cache hits) | {results["api_calls_saved"]} |
| **Cost saved** (same token profile per request) | **{results["savings_pct"]:.1f}%** |

## Latency

| Path | p50 (ms) | p95 (ms) |
|---|---|---|
| Raw OpenAI SDK | {results["raw_p50"]:.3f} | {results["raw_p95"]:.3f} |
| CAELUM, cache miss | {results["miss_p50"]:.3f} | {results["miss_p95"]:.3f} |
| CAELUM, cache hit | {results["hit_p50"]:.3f} | {results["hit_p95"]:.3f} |

**CAELUM overhead on cache miss: p50 {overhead_p50:+.3f} ms, p95 {overhead_p95:+.3f} ms**
— target p95 < 5 ms, hard ceiling 10 ms: **{verdict}**.

A cache hit skips the network entirely: in production, a real API call typically
takes hundreds of milliseconds, while a hit takes about {results["hit_p50"]:.2f} ms.
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
