# Integration tests — real APIs (2026-11-12)

Models: OpenAI `gpt-4o-mini`, Anthropic `claude-haiku-5-5`, Gemini `gemini-3.1-flash-lite`, Mistral `mistral-small-latest` — `max_tokens=16`.

| Test | Result |
| --- | --- |
| OpenAI exact cache (20 calls) | 10 hits, $0.000029 saved, p95 1319 ms |
| Anthropic exact cache (20 calls) | 10 hits, $0.000034 saved, p95 1125 ms |
| OpenAI streaming | 35 characters streamed |
| Semantic cache, with gpt-4o-mini verifier (floor 0.60) | 37/50 paraphrases reused, 0/50 false positives, 129 checks, verifier $0.008764 |
| Semantic cache, embeddings only (0.97) | 0/50 paraphrases reused, 0/50 false positives, 0 checks, verifier $0.000000 |
| Router (3 profiles) | rewrite -> claude-haiku-5-5, long -> None, plain -> None |
| Fallback (OpenAI unreachable) | answered by claude-haiku-5-5 |
| Anthropic + OpenAI embedder | cache type: None |
| Gemini exact cache (11 calls, sync + async) | 6 hits, $0.000023 saved, cost $0.000019, p95 1605 ms |
| OpenAI-compatible: mistralai/mistral-small-24b-instruct-2501 (6 calls) | 3 hits, 0 retries, p95 1467 ms |
| OpenAI-compatible: deepseek/deepseek-v4-flash (6 calls) | 3 hits, 0 retries, p95 1285 ms |
| OpenAI-compatible: meta-llama/llama-3.1-8b-instruct (6 calls) | 3 hits, 0 retries, p95 1287 ms |
| OpenAI-compatible: qwen/qwen3.7-flash (6 calls) | 3 hits, 1 retries, p95 3062 ms |
| OpenAI-compatible: google/gemma-3-4b-it (6 calls) | 3 hits, 0 retries, p95 1420 ms |
| OpenAI-compatible: x-ai/grok-4.3 (6 calls) | 3 hits, 0 retries, p95 7000 ms |
| OpenAI-compatible: microsoft/phi-4 (6 calls) | 3 hits, 0 retries, p95 942 ms |
| OpenAI-compatible: cohere/command-r7b-12-2024 (6 calls) | 3 hits, 0 retries, p95 743 ms |
| OpenAI-compatible: amazon/nova-micro-v1 (6 calls) | 3 hits, 0 retries, p95 1377 ms |
| OpenAI-compatible: moonshotai/kimi-k2.5 (6 calls) | 3 hits, 0 retries, p95 4381 ms |
| OpenAI-compatible: z-ai/glm-5.3-flash (6 calls) | 3 hits, 0 retries, p95 3023 ms |
| OpenAI-compatible: nvidia/nemotron-3.5-lightning (6 calls) | 3 hits, 0 retries, p95 1222 ms |
| OpenAI-compatible: ibm-granite/granite-4.2-8b (6 calls) | 3 hits, 0 retries, p95 1334 ms |
| OpenAI-compatible: minimax/minimax-m3.1-flash (6 calls) | 3 hits, 0 retries, p95 2586 ms |
| OpenAI-compatible: upstage/solar-mini4 (6 calls) | 3 hits, 0 retries, p95 3265 ms |
| OpenAI-compatible: perplexity/sonar (6 calls) | 3 hits, 0 retries, p95 2108 ms |
| OpenAI-compatible streaming (Mistral via OpenRouter) | 14 characters |
