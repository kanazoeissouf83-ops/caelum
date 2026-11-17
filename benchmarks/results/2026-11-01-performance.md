# Benchmark — CAELUM latency overhead (2026-11-01)

Overhead measured **per request** as total time minus the time spent inside the provider
SDK call (500 cache-miss requests per row). Fake OpenAI server with a simulated
20 ms provider latency (real calls are I/O-bound: hundreds of
milliseconds). Python 3.14.4, Windows AMD64.

| Configuration | Mode | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|---|
| wrapper only | sequential | 0.110 | **0.207** | 0.329 | 1.364 |
| wrapper only | 8 threads | 0.069 | **0.140** | 0.206 | 0.277 |
| cache=True (miss) | sequential | 0.516 | **0.867** | 1.038 | 1.679 |
| cache=True (miss) | 8 threads | 0.207 | **2.149** | 10.490 | 17.414 |
| cache + router + retry | sequential | 0.495 | **0.895** | 1.230 | 1.953 |
| cache + router + retry | 8 threads | 0.224 | **3.302** | 6.110 | 10.483 |

**Worst p95 overhead: 3.302 ms** — target < 5 ms, ceiling 10 ms: **PASS**.

Note: with a zero-latency fake server, 8 CPU-bound threads contend for the Python GIL and
every SQLite call has to wait for it again; that synthetic setup inflates tail latency
(around +50 ms at p95) without reflecting real, network-bound workloads.
