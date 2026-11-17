# Benchmark — exact cache (2026-11-01)

Workload: 1000 chat requests, 30% duplicates
(700 unique questions), fake in-memory OpenAI server
(no network: latencies below measure SDK + CAELUM work only).

## Savings

| Metric | Value |
|---|---|
| Requests | 1000 |
| API calls made | 700 |
| API calls saved (cache hits) | 300 |
| **Cost saved** (same token profile per request) | **30.0%** |

## Latency

| Path | p50 (ms) | p95 (ms) |
|---|---|---|
| Raw OpenAI SDK | 0.593 | 1.240 |
| CAELUM, cache miss | 0.712 | 1.486 |
| CAELUM, cache hit | 0.061 | 0.151 |

**CAELUM overhead on cache miss: p50 +0.120 ms, p95 +0.245 ms**
— target p95 < 5 ms, hard ceiling 10 ms: **PASS**.

A cache hit skips the network entirely: in production, a real API call typically
takes hundreds of milliseconds, while a hit takes about 0.06 ms.
