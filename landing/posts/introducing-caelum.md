---
title: "Introducing CAELUM: stop paying twice for the same AI answer"
description: "An open-source Python library that adds caching, fallback and cost metrics to the OpenAI, Anthropic, Gemini and Mistral SDKs, by changing one import."
author: "Issouf KANAZOE"
---

AI APIs are billed per token, and many applications ask the same questions again and again:
the same support question from different customers, the same document summarized twice,
the same test suite run every day. Each time, you pay for an answer you already have.

CAELUM is a small open-source Python library that fixes that, and a few other things, without
changing how you write code.

## One import

```python
from caelum import OpenAI   # instead of: from openai import OpenAI

client = OpenAI(cache=True, retry=True)
```

Everything else stays the same: your code, your types, your API keys. CAELUM wraps the
official OpenAI, Anthropic, Google Gemini and Mistral SDKs instead of replacing them, so new
SDK features work on day one.

## What you get

- **Exact cache.** Identical requests are answered from a local SQLite file, in a fraction of
  a millisecond, for free.
- **Semantic cache (opt-in).** A rephrased question ("Can I get my money back?" /
  "Do you offer refunds?") can reuse a stored answer, but only after a small model confirms
  that both questions really ask the same thing.
- **Retries and fallback.** When a provider is down or rate-limits you, CAELUM retries with
  backoff, then falls back to the model you chose on another provider, or to a free local
  model (Ollama, llama.cpp, vLLM, LM Studio).
- **Routing by config.** Send short, simple requests to a cheaper model from a YAML, JSON or
  dict file.
- **Cost metrics.** `client.metrics.print_summary()` shows requests, cache hits, money saved
  and money spent.

Everything is opt-in, runs on your machine, and nothing is sent to us. You keep paying your
providers directly: CAELUM never resells tokens.

## Measured, not promised

- **30% of calls saved** on a benchmark where 30% of requests repeat. Your savings depend on
  how often your own requests repeat.
- **Under 1 ms** of overhead at p95 on cache misses.
- **Semantic cache: 37 of 50 rephrasings reused, 0 of 50 look-alike questions answered
  wrongly** in a real API test. Embeddings alone could not do this safely: "convert miles to
  km" and "convert km to miles" look almost identical to them.
- **Multi-tenant safe:** cached answers are isolated per API key, and five prompt-injection
  attempts against the semantic-cache judge were all rejected.
- **369 automated tests**, plus real-API tests on OpenAI, Anthropic, Gemini and 16 model
  families through OpenRouter.

All benchmark scripts and raw results are in the repository, so you can check them on your
own traffic.

## Why I built it

I'm Issouf, based between Burkina Faso and Ghana. I come from the product and business side,
not from a career in engineering. I started CAELUM because AI API costs are a real barrier for
small teams, and even more for teams paying in currencies that are weak against the dollar.
Paying twice for the same answer felt like the first thing to fix.

## What's next

1. Your feedback, and version 0.2.
2. Routing and fallback for Gemini and Mistral.
3. CAELUM Cloud: an optional hosted dashboard for teams, with one cache shared by all your
   servers. The library stays open source and works fully without it.

Try it, and tell us what's missing:

```bash
pip install "caelum-ai[openai]"
```
