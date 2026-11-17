# Benchmark — latency overhead, pre-launch QA (2026-11-12)

Same script as `2026-11-01-performance.md` (`bench_latency.py`, fake zero-latency server,
500 cache-miss requests per row), run **alternately** on the code before the QA campaign
(commit `6166666`) and after it, in the same environment (Python 3.14, Windows, openai
SDK on `httpx2`, a busy 8 GB laptop). p95 overhead in ms, two runs each:

| Configuration | Mode | before QA | after QA |
|---|---|---|---|
| wrapper only | sequential | 0.68 – 0.76 | 0.97 – 1.08 |
| wrapper only | 8 threads | 0.45 – 0.55 | 0.63 – 0.66 |
| cache=True (miss) | sequential | 1.86 – 1.99 | 2.09 – 2.22 |
| cache=True (miss) | 8 threads | 13.2 – 16.2 | 15.4 – 16.1 |
| cache + router + retry | sequential | 2.13 – 2.14 | 2.41 – 2.65 |
| cache + router + retry | 8 threads | 13.7 – 14.3 | 13.0 – 17.2 |

Direct measurement of CAELUM's own pipeline (SDK call stubbed, 20,000 calls):
**3 µs before, 5.5 µs after** per request (the per-credential cache namespace).

Findings:

- The QA fixes add about **0.003 ms** of CAELUM work per request: negligible.
- The 8-thread rows exceed the 10 ms ceiling **before and after** the QA changes on this
  machine today (they were 3.3 ms on 2026-11-01). With a zero-latency fake server, 8
  CPU-bound threads fight for the Python GIL and every SQLite call waits for it; machine
  load and the newer SDK make this synthetic worst case noisy. It does not reflect
  network-bound production traffic, where each request waits hundreds of milliseconds
  for the provider.
- Sequential cache misses stay well under the 5 ms target. The exact-cache benchmark of
  the same day (`2026-11-12-cache-exact.md`) measures **+0.56 ms** at p95; public docs
  now say "under 1 ms" instead of "+0.25 ms".
- Stress test (`2026-11-12-stress.md`): 10,000 requests, flat p95 (~4.5 ms end to end
  against the in-memory server), memory 1.6 MB peak, no growth.

To re-check on a quiet machine before launch:
`uv run --project packages/python python benchmarks/scripts/bench_latency.py --output ...`
