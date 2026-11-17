# Integration tests — real APIs (2026-11-02)

Models: OpenAI `gpt-4o-mini`, Anthropic `claude-haiku-5-5`, Gemini `gemini-3.1-flash-lite`, Mistral `mistral-small-latest` — `max_tokens=16`.

| Test | Result |
| --- | --- |
| OpenAI exact cache (20 calls) | 10 hits, $0.000029 saved, p95 831 ms |
| Anthropic exact cache (20 calls) | 10 hits, $0.000034 saved, p95 996 ms |
| OpenAI streaming | 31 characters streamed |
| Router (3 profiles) | rewrite -> claude-haiku-5-5, long -> None, plain -> None |
| Fallback (OpenAI unreachable) | answered by claude-haiku-5-5 |
| Anthropic + OpenAI embedder | cache type: None |
| Gemini exact cache (11 calls, sync + async) | 6 hits, $0.000023 saved, cost $0.000019, p95 2382 ms |
| OpenAI-compatible: mistralai/mistral-small-24b-instruct-2501 (6 calls) | 3 hits, 0 retries, p95 1149 ms |
| OpenAI-compatible: deepseek/deepseek-v4-flash (6 calls) | 3 hits, 0 retries, p95 1323 ms |
| OpenAI-compatible: meta-llama/llama-3.1-8b-instruct (6 calls) | 3 hits, 0 retries, p95 1926 ms |
| OpenAI-compatible: qwen/qwen3.7-flash (6 calls) | 3 hits, 0 retries, p95 2075 ms |
| OpenAI-compatible: google/gemma-3-4b-it (6 calls) | 3 hits, 0 retries, p95 1517 ms |
| OpenAI-compatible streaming (Mistral via OpenRouter) | 14 characters |
| OpenAI-compatible: x-ai/grok-4.3 (6 calls) | 3 hits, 0 retries, p95 6522 ms |
| OpenAI-compatible: microsoft/phi-4 (6 calls) | 3 hits, 0 retries, p95 1229 ms |
| OpenAI-compatible: cohere/command-r7b-12-2024 (6 calls) | 3 hits, 0 retries, p95 1099 ms |
| OpenAI-compatible: amazon/nova-micro-v1 (6 calls) | 3 hits, 0 retries, p95 1309 ms |
| OpenAI-compatible: moonshotai/kimi-k2.5 (6 calls) | 3 hits, 0 retries, p95 3780 ms |
| OpenAI-compatible: z-ai/glm-5.3-flash (6 calls) | 3 hits, 0 retries, p95 5637 ms |
| OpenAI-compatible: nvidia/nemotron-3.5-lightning (6 calls) | 3 hits, 0 retries, p95 2317 ms |
| OpenAI-compatible: ibm-granite/granite-4.2-8b (6 calls) | 3 hits, 0 retries, p95 1214 ms |
| OpenAI-compatible: minimax/minimax-m3.1-flash (6 calls) | 3 hits, 0 retries, p95 1703 ms |
| OpenAI-compatible: bytedance-seed/seed-1.6-flash (6 calls) | 3 hits, 0 retries, p95 1427 ms |
| OpenAI-compatible: perplexity/sonar (6 calls) | 3 hits, 0 retries, p95 1595 ms |
| OpenAI-compatible: upstage/solar-mini4 (6 calls) | 3 hits, 0 retries, p95 2436 ms |
| Semantic cache, with gpt-4o-mini verifier (floor 0.60) | 38/50 paraphrases reused, 0/50 false positives, 128 checks, verifier $0.007133 |
| Semantic cache, embeddings only (0.97) | 0/50 paraphrases reused, 0/50 false positives, 0 checks, verifier $0.000000 |
| Local: local server down -> real OpenAI fallback | answered by gpt-4o-mini |
