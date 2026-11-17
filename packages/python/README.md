# CAELUM Python SDK

The control plane for AI inference — drop-in caching, routing, observability and fallback
for LLM APIs.

> **Status: pre-release.** The API may still change before 1.0.

Change one import and get, all opt-in:

- **exact and semantic cache** (local SQLite file; rephrased questions are reused only
  after a small model confirms they ask the same thing);
- **routing** to the models you choose, from a dict, JSON or YAML file;
- **retries** with backoff and **fallback** to another provider or a local model
  (Ollama, llama.cpp, vLLM, LM Studio);
- **metrics**: calls, tokens, cache hits, latency, estimated cost;
- `caelum cache stats` to see what your cache holds.

Works with the official OpenAI, Anthropic, Google Gen AI (Gemini) and Mistral SDKs, with
their types unchanged. Python 3.10+, tested on Windows and Linux.

## Installation

```bash
pip install "caelum-ai[openai]"      # with the OpenAI SDK
pip install "caelum-ai[anthropic]"   # with the Anthropic SDK
pip install "caelum-ai[gemini]"      # with the Google Gen AI SDK (Gemini)
pip install "caelum-ai[mistral]"     # with the Mistral SDK
pip install "caelum-ai[rich]"        # prettier metrics output
pip install "caelum-ai[yaml]"        # routing rules in YAML files
pip install "caelum-ai[fast]"        # faster semantic cache on large caches (numpy)
```

The package is published as `caelum-ai` and imported as `caelum`:

```python
from caelum import OpenAI  # instead of: from openai import OpenAI

client = OpenAI(cache=True, retry=True)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
client.metrics.print_summary()
```

Documentation: [github.com/kanazoeissouf83-ops/caelum/tree/main/docs](https://github.com/kanazoeissouf83-ops/caelum/tree/main/docs).

Provider SDKs are optional extras: the core package stays lightweight.

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd packages/python
uv sync --all-extras          # create .venv and install everything (incl. dev tools)
uv run pytest                 # tests + coverage
uv run ruff format .          # format
uv run ruff check .           # lint
uv run pyright                # type check (strict)
```

Integration tests calling real APIs are marked `integration` and excluded by default:

```bash
uv run pytest -m integration
```

## License

Apache 2.0 — see [LICENSE](https://github.com/kanazoeissouf83-ops/caelum/blob/main/LICENSE).
