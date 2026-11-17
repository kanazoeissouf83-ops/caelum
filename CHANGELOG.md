# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-18

First public release. The package is published as **`caelum-ai`** and imported as `caelum`.

### Security

- Cached answers are now isolated per credential (API key, organization, project):
  in a multi-tenant app, a client using one customer's key is never served another
  customer's cached answer. Only a short hash of the key is used; the key is never
  written to the cache file. Changing `client.api_key` switches to that key's cache
  immediately.
- The semantic-cache verifier receives both questions as escaped JSON and is told to
  ignore instructions inside them (prompt-injection hardening); `caching.mdx` documents
  the remaining risk for multi-user apps.
- The cache folder and file are created private to the current user on Linux/macOS
  (`0700` / `0600`).
- GitHub Actions are pinned by commit SHA; the CLA workflow's write permissions are
  scoped to its job; CI installs exactly the locked dependencies (`uv sync --locked`).

### Changed

- **CAELUM Cloud plans are measured in questions per month**, not provider tokens: Free
  10,000, Starter 1,000,000 (every question counts, answered from the cache or not).
  Cost per end customer, budget alerts, 90 days of history and the shared repetition
  counter are Starter features; free accounts see 7 days of history. The SDK stops asking
  the shared counter for an hour after a "plan_required" answer and keeps its local one.
- `cache_ttl=None` now means the 30-day maximum instead of "never expires", and lifetimes
  under 1 minute are refused. CAELUM Cloud entries are now filed per end customer (a short
  hash, never the name): entries written by earlier versions are no longer found.
- **Conversations are never cached**: a request with an earlier AI answer, a tool result
  or more than one user message no longer reads or writes any cache (a follow-up such as
  "and to Ouaga?" depends on what came before).
- Answers are no longer shared between end customers (`caelum.customer()`), and existing
  cache entries are not reused after the upgrade: the cache key format version is now 2.
- Removed the unused core dependencies `httpx` and `tiktoken`: `pip install caelum-ai`
  now installs nothing besides CAELUM itself.

### Fixed

- The request journal no longer tries to write after its file was closed at shutdown.
- A corrupted cache file no longer leaves an open database connection behind.
- The local cache file no longer grows forever: expired answers (never served) are now
  removed from disk automatically, at most every 10 minutes, in the background.
- CAELUM Cloud far from Cloudflare's database (found by a real test from West Africa):
  connections are now kept alive and reused (a lookup takes ~0.3 s instead of ~0.6 s),
  a slow lookup no longer pauses the uploads of metrics and shared-cache answers, answers
  are uploaded at once instead of within 5 s, and the last metrics of a short-lived
  program are no longer lost at exit.
- Fork safety: worker processes (gunicorn, uwsgi, multiprocessing) reopen their own cache
  connections instead of reusing the parent's, so their cache writes are no longer lost.
- With `retry=True`, the fallback/routing clients created by CAELUM no longer retry on
  their own as well (up to 12 calls instead of 4 during an outage).
- Malformed tool-call arguments in the history (e.g. a tool call truncated by
  `max_tokens`) no longer crash routed or fallback requests: the route is skipped.
- Translation between providers no longer loses information: `extra_body` /
  `extra_query` are never sent to another provider, the `is_error` flag of tool results
  and OpenAI refusals are kept, and empty assistant turns are refused instead of
  producing a request Claude rejects.
- `anthropic-beta` / `openai-beta` headers are now part of the cache key (they can
  change the answer).
- In async code, semantic-cache search and storage run in a worker thread instead of
  blocking the event loop.
- `close()` also closes the fallback/routing clients CAELUM created; writes after
  `close()` are dropped instead of piling up in memory.
- The semantic verifier has a timeout (`LLMVerifier(timeout=10.0)` by default) and no
  SDK retries: a slow judge can no longer block a request for minutes. Only a plain
  "YES" counts as approval ("YES, but actually NO" is a NO).
- Tool-call arguments that are valid JSON but not an object (`[1, 2]`, `null`) are not
  sent to Claude (the route or fallback tier is skipped).
- Wrong option values are rejected at construction with a clear message: non-numeric or
  boolean `cache_ttl` / `semantic_threshold`, empty `cache_path`, a `cache_store` that is
  not a `CacheStore`, a `router_config` of the wrong type, `provider_keys` that is not a
  dict, and sync `provider_clients` given to an async client (or the reverse). Passing
  `cache_path` without `cache=True` logs a warning.
- Router validation is stricter: empty `match`, empty lists or empty strings (which
  would match every request) are refused, and errors name the faulty key.
- `caelum cache stats` no longer crashes on a hand-edited row with a non-numeric expiry.
- `client.metrics.export_json()` creates missing folders.

- An unusable cache file (read-only, corrupted, or a directory) no longer crashes the
  client constructor: the cache is disabled with a warning and the API is called
  normally, as documented.
- Install commands use double quotes everywhere (`pip install "caelum-ai[openai]"`),
  which work in cmd.exe, PowerShell and Unix shells alike.
- Costs below one cent are no longer shown as `$ 0.00` in `print_summary()` and
  `caelum cache stats` (a request to a small model costs a fraction of a cent).
- The "missing extra" error now names the real package (`google-genai`, `mistralai`).
- The pretty (rich) metrics table now mentions requests whose cost is unknown.
- A request holding values that cannot be serialized (e.g. Python functions as tools,
  or a class as response schema) is no longer a crash: it is simply not cached.

### Added

- **Shared admission counter** (CAELUM Cloud): with a CAELUM key, repetitions are counted
  for the whole account, so a question answered once on each of two servers is admitted.
  Counted after the answer is returned, atomically (new `POST /v1/admission` route, D1
  table storing only the question's hash, a counter and dates), with expiry and a cap
  per end customer; falls back to the local counter when the Cloud is unreachable. New
  metrics `admission_shared_*`, `client.shared_admission_pending()`, and the journal's
  `admission_counter` (`local` / `shared`).
- **Net savings**: the request journal records, per request, the judge calls and their
  cost, the embedding cost (real tokens or the new `judge_call_cost` / `embedding_call_cost`
  estimates) and the time they add. `client.journal.stats()` reports gross savings,
  verification cost, net savings, cache vs direct AI response time, the break-even hit
  rate, and plain-French alerts when the net savings are negative or the cache is slower
  than the AI.
- **Threshold per request**: `semantic_threshold=` (0.80 to 0.999), optional
  `semantic_gray_threshold=` (strictly below), or `semantic_strict=True` (0.999) on one
  request. A client-level safety floor (`semantic_threshold_floor`) can never be
  undercut: a lower request threshold is raised to it. Priority: cache_bypass, then
  exclusions, then request, then end customer, then default. The journal records the
  threshold used and its source.
- **Lifetime per end customer** (`cache_customer_ttl`), bounded to 1 minute .. 30 days
  with clear French / English error messages. The expiry is `cached at + lifetime` both
  locally and in CAELUM Cloud; a shortened lifetime applies to answers already cached, a
  longer one does not extend them.
- **Clearing a customer's cache**: `client.clear_cache(customer, older_than=None)` and
  `caelum cache clear --customer ... [--older-than ...]` delete one end customer's answers
  locally and in CAELUM Cloud (new `DELETE /v1/cache` route); each clearing is noted in the
  request journal.
- **False-positive test set**: 138 question pairs (40 new French / West African ones) with
  recorded real embeddings and judge verdicts, replayed offline on every test run through
  the real pipeline; the false-positive and reuse rates are printed at the end of the run
  and any false positive fails the suite. Measured: 0 % false positives with the judge
  (72.5 % of paraphrases reused), 0 % with embeddings only at 0.97 (no reuse), 3 % at
  0.90, 19.4 % at 0.85. `benchmarks/scripts/record_semantic_pairs.py` re-records them.
- **Per-request journal** (`client.journal`, local SQLite file `~/.caelum/journal.sqlite`,
  never sent to CAELUM Cloud): one line per request with outcome (hit, gray, miss,
  excluded, bypassed), similarity score, cache entry, cost avoided (real token cost or the
  `avoided_call_cost` estimate, $0.002 by default, per end customer with
  `customer_avoided_call_cost`) and response time. Excluded and bypassed requests are
  logged without any text. 30-day retention (`journal_retention_days`), 100,000-line cap,
  background writes; `client.journal.clear()` and `caelum journal clear` erase it.
  `client.journal.stats(customer)` gives requests, hit rate, outcomes, savings and the 20
  lowest-score hits. New metric: `journal_errors`.
- **Automatic exclusions**: a question containing an order or tracking number, a phone
  number (+226, +233, +225...), an email address, an amount (FCFA, XOF, GHS, $, €...), a
  relative or precise date or time, or payment / identity data is never read from or
  written to the cache, never counted for admission, and its text is kept nowhere.
  Categories can be switched off globally (`cache_exclusions`) or per end customer
  (`cache_customer_exclusions`); custom regular expressions can be added
  (`cache_exclude_patterns`). New metrics: `excluded_requests` and `exclusions` per
  category.
- **Frequency admission** (`cache_admission`, default 2, per end customer with
  `cache_customer_admission`): an answer is cached only after the API has answered the
  same question (or, with the semantic cache, a question with the same meaning) N times.
  Questions asked once are never stored. **Effect on savings: with N=2, savings start with
  the 3rd request of a question, not the 2nd.** Use `cache_admission=1` for the previous
  behaviour. New metrics: `admission_pending`, `admission_admitted`, `admission_refused`.
- Cache keys now contain the **end customer**, the **language** (`caelum.language()` or
  detected), the **AI model** and a fingerprint of the **system instructions**.
- `cache_bypass=True` (request argument) and `caelum.cache_bypass()` (block) skip every
  cache for a request.
- Semantic cache **gray zone**: a match between `semantic_gray_threshold` (default
  `semantic_threshold - 0.05`) and `semantic_threshold` calls the API and replaces the
  stored entry with the fresh answer. Both bounds can be set per end customer
  (`semantic_customer_thresholds`, or `set_thresholds()` at runtime). New metric:
  `semantic_refreshes`.
- CAELUM Cloud connection (optional): `caelum_key="sk-caelum-..."` (or `$CAELUM_API_KEY`)
  on every client sends usage metrics (never prompts or provider keys) to the CAELUM
  Cloud dashboard and, with `cache=True`, adds a shared cache behind the local one, so
  that every server and serverless instance reuses the same answers. `caelum.customer()`
  attributes requests to an end customer (cost per customer). The Cloud is never on the
  critical path: 1 s max per lookup, background uploads, circuit breaker, standard
  library only. See `docs/cloud.mdx`.
- Contributor License Agreement (`CLA.md`, signed once per contributor through a bot
  on pull requests), trademark policy (`TRADEMARKS.md`) and `NOTICE` file (shipped in
  the package). Secret scanning (gitleaks) of the whole history on every push.
- `caelum cache stats` command (also `python -m caelum cache stats`): entries (exact and
  semantic), expired entries, size on disk, oldest/newest entry, and the estimated value
  of the cached answers per model. `--path` and `--json` options. Read-only.
- Local models as fallback and routing targets: `ollama`, `llamacpp`, `vllm` and
  `lmstudio` providers (OpenAI-compatible, no key), e.g.
  `fallback=["ollama:qwen3:1.7b"]`. Local answers cost $0 in the metrics. A client whose
  `base_url` points to `localhost` is recognized as local: its requests are free and an
  `openai:` fallback tier reaches the real OpenAI API. OpenAI-format clients can stream
  from a local fallback. Addresses configurable with `CAELUM_<NAME>_BASE_URL` (or
  `OLLAMA_HOST`).
- Semantic cache verification: before reusing an answer, a small model
  (`caelum.LLMVerifier`, default `gpt-4o-mini` through the OpenAI client) checks that both
  questions ask for exactly the same thing, in both orders. The candidate threshold drops
  to 0.60 with a verifier. On our calibration set: 74-78% of paraphrases reused (vs 0-2%
  with embeddings alone) and no wrong answer served. New metrics:
  `semantic_verifications`, `semantic_rejections`, `verifier_cost_usd`. Opt out with
  `semantic_verifier=False`. Semantic entries now store their question text.
- Mistral support: `caelum.Mistral` wraps `mistralai.client.Mistral`; exact cache,
  retries and metrics on `chat.complete` and `chat.complete_async`. Extra:
  `caelum-ai[mistral]`. Mistral prices in `caelum.pricing`.
- Docs: using CAELUM with OpenAI-compatible providers (DeepSeek, Groq, xAI, Ollama).
- Google Gemini support: `from caelum import genai` mirrors `google.genai`; `genai.Client`
  adds the exact cache, retries and metrics to `models.generate_content` (sync and `aio`).
  Extra: `caelum-ai[gemini]`. Gemini prices in `caelum.pricing`.
- Drop-in wrappers `caelum.OpenAI`, `caelum.AsyncOpenAI`, `caelum.Anthropic` and
  `caelum.AsyncAnthropic`: same arguments, methods, objects and static types as the
  official SDKs.
- Exact response cache (`cache=True`): local SQLite storage, JSON values (no pickle),
  configurable TTL, never breaks a request on storage errors.
- Semantic cache (`semantic_cache=True`): reuses answers to paraphrased questions asked
  in exactly the same context; strict default threshold (0.97); local vector store with
  optional numpy acceleration (`caelum-ai[fast]`).
- Local metrics (`client.metrics`): requests, cache hits, failures, retries, estimated
  cost and savings, latency percentiles; `print_summary()`, `summary()`, `export_json()`.
- Price table for cost estimates (`caelum.pricing`) with `register_price()`.
- Opt-in retries (`retry=True` or `RetryPolicy`) for 429/5xx/529 and network errors,
  honouring `Retry-After`, re-raising the original SDK exception.
- Rule-based routing (`router_config=`: dict, JSON or YAML via `caelum-ai[yaml]`) to the
  cheapest capable model, including across providers.
- Multi-provider fallback (`fallback=["anthropic:claude-sonnet-5-5"]`) on transient
  errors, with faithful OpenAI ↔ Anthropic request/response translation.
- `caelum.cache_hit()`, `caelum.cache_type()` and `caelum.routed_to()` helpers.

### Performance

- Cache writes happen in a background thread and reads use per-thread connections:
  worst measured p95 overhead 3.3 ms with 8 concurrent threads (target < 5 ms).
