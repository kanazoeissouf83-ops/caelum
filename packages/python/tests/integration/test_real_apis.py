"""Real-API checks of every CAELUM promise (cache, semantic cache, routing, fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import Anthropic, OpenAI, OpenAIEmbedder
from tests.integration.conftest import (
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    MAX_TOKENS,
    MISTRAL_MODEL,
    OPENAI_MODEL,
    record,
    user,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_openai_exact_cache_savings(openai_key: str, tmp_path: Path) -> None:
    client = OpenAI(api_key=openai_key, cache=True, cache_path=tmp_path / "c.sqlite")
    questions = [f"Reply with the number {i} only." for i in range(10)]
    for question in questions * 2:  # 20 calls, 10 duplicates
        client.chat.completions.create(
            model=OPENAI_MODEL, messages=user(question), max_tokens=MAX_TOKENS
        )
    s = client.metrics.summary()
    record(
        "OpenAI exact cache (20 calls)",
        f"{s['cache_hits']} hits, ${s['cost_saved_usd']:.6f} saved, "
        f"p95 {s['latency_p95_ms']:.0f} ms",
    )
    assert s["cache_hits"] == 10
    assert s["api_calls"] == 10


def test_anthropic_exact_cache_savings(anthropic_key: str, tmp_path: Path) -> None:
    client = Anthropic(api_key=anthropic_key, cache=True, cache_path=tmp_path / "c.sqlite")
    questions = [f"Reply with the number {i} only." for i in range(10)]
    for question in questions * 2:
        client.messages.create(
            model=ANTHROPIC_MODEL, messages=user(question), max_tokens=MAX_TOKENS
        )
    s = client.metrics.summary()
    record(
        "Anthropic exact cache (20 calls)",
        f"{s['cache_hits']} hits, ${s['cost_saved_usd']:.6f} saved, "
        f"p95 {s['latency_p95_ms']:.0f} ms",
    )
    assert s["cache_hits"] == 10


def test_openai_streaming_is_untouched(openai_key: str) -> None:
    client = OpenAI(api_key=openai_key)
    stream = client.chat.completions.create(
        model=OPENAI_MODEL, messages=user("Say hi."), max_tokens=MAX_TOKENS, stream=True
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    record("OpenAI streaming", f"{len(text)} characters streamed")
    assert text


def _semantic_run(client: OpenAI) -> tuple[int, int, int, int]:
    """Ask every calibration pair through one shared cache; return reuse and false hits."""
    data: dict[str, Any] = json.loads(
        (FIXTURES / "semantic_pairs.json").read_text(encoding="utf-8")
    )
    similar: list[list[str]] = data["similar"]
    different: list[list[str]] = data["different"]

    def ask(question: str) -> Any:
        return client.chat.completions.create(
            model=OPENAI_MODEL, messages=user(question), max_tokens=MAX_TOKENS
        )

    hits = false_positives = 0
    for first, second in similar:
        ask(first)
        hits += caelum.cache_type(ask(second)) == "semantic"
    for first, second in different:
        ask(first)
        false_positives += caelum.cache_hit(ask(second))
    return hits, len(similar), false_positives, len(different)


@pytest.mark.parametrize("verified", [True, False], ids=["verified", "embeddings-only"])
def test_semantic_cache_hit_rate(verified: bool, openai_key: str, tmp_path: Path) -> None:
    client = OpenAI(
        api_key=openai_key,
        semantic_cache=True,
        semantic_verifier=None if verified else False,
        cache_path=tmp_path / "s.sqlite",
    )
    hits, n_similar, false_positives, n_different = _semantic_run(client)
    s = client.metrics.summary()
    label = "with gpt-4o-mini verifier (floor 0.60)" if verified else "embeddings only (0.97)"
    record(
        f"Semantic cache, {label}",
        f"{hits}/{n_similar} paraphrases reused, {false_positives}/{n_different} false "
        f"positives, {s['semantic_verifications']} checks, verifier ${s['verifier_cost_usd']:.6f}",
    )
    assert false_positives == 0  # zero false positives is the priority


def test_router_profiles(openai_key: str, anthropic_key: str) -> None:
    config: dict[str, Any] = {
        "routes": [
            {
                "name": "rewrite-to-claude",
                "match": {"system_contains": ["rewrite"]},
                "target": ANTHROPIC_MODEL,
                "provider": "anthropic",
            },
            {
                "name": "long-prompts",
                "match": {"messages_min_tokens": 200},
                "target": OPENAI_MODEL,
                "provider": "openai",
            },
        ]
    }
    client = OpenAI(
        api_key=openai_key, router_config=config, provider_keys={"anthropic": anthropic_key}
    )
    rewrite = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "Rewrite in plain English."}, *user("Utilize it.")],
        max_tokens=MAX_TOKENS,
    )
    long_prompt = client.chat.completions.create(
        model=OPENAI_MODEL, messages=user("word " * 300 + "Say ok."), max_tokens=MAX_TOKENS
    )
    plain = client.chat.completions.create(
        model=OPENAI_MODEL, messages=user("Say ok."), max_tokens=MAX_TOKENS
    )
    record(
        "Router (3 profiles)",
        f"rewrite -> {caelum.routed_to(rewrite)}, long -> {caelum.routed_to(long_prompt)}, "
        f"plain -> {caelum.routed_to(plain)}",
    )
    assert caelum.routed_to(rewrite) == ANTHROPIC_MODEL
    assert rewrite.choices[0].message.content
    assert caelum.routed_to(plain) is None


def test_fallback_when_primary_is_unreachable(openai_key: str, anthropic_key: str) -> None:
    client = OpenAI(
        api_key=openai_key,
        base_url="http://127.0.0.1:9/v1",  # nothing listens here: connection error
        max_retries=0,
        timeout=5,
        fallback=[f"anthropic:{ANTHROPIC_MODEL}"],
        provider_keys={"anthropic": anthropic_key},
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=user("Say ok."), max_tokens=MAX_TOKENS
    )
    record("Fallback (OpenAI unreachable)", f"answered by {caelum.routed_to(response)}")
    assert caelum.routed_to(response) == ANTHROPIC_MODEL
    assert response.choices[0].message.content


def test_openai_embedder_for_anthropic(openai_key: str, anthropic_key: str, tmp_path: Path) -> None:
    client = Anthropic(
        api_key=anthropic_key,
        semantic_cache=True,
        semantic_embedder=OpenAIEmbedder(api_key=openai_key),
        cache_path=tmp_path / "s.sqlite",
    )
    client.messages.create(
        model=ANTHROPIC_MODEL, messages=user("How do I reset my password?"), max_tokens=MAX_TOKENS
    )
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        messages=user("How can I reset my password?"),
        max_tokens=MAX_TOKENS,
    )
    record("Anthropic + OpenAI embedder", f"cache type: {caelum.cache_type(response)}")
    assert response.content


def test_gemini_exact_cache_sync_and_async(gemini_key: str, tmp_path: Path) -> None:
    import asyncio

    from caelum import genai

    client = genai.Client(api_key=gemini_key, cache=True, cache_path=tmp_path / "c.sqlite")
    config = genai.types.GenerateContentConfig(max_output_tokens=MAX_TOKENS)
    questions = [f"Reply with the number {i} only." for i in range(5)]
    for question in questions * 2:  # 10 calls, 5 duplicates
        response = client.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=GEMINI_MODEL, contents=question, config=config
        )
        assert response.candidates
    again = asyncio.run(
        client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=GEMINI_MODEL, contents=questions[0], config=config
        )
    )
    s = client.metrics.summary()
    record(
        "Gemini exact cache (11 calls, sync + async)",
        f"{s['cache_hits']} hits, ${s['cost_saved_usd']:.6f} saved, "
        f"cost ${s['cost_usd']:.6f}, p95 {s['latency_p95_ms']:.0f} ms",
    )
    assert caelum.cache_hit(again)
    assert s["cache_hits"] == 6
    assert s["api_calls"] == 5


def test_mistral_exact_cache_sync_and_async(mistral_key: str, tmp_path: Path) -> None:
    import asyncio

    from caelum import Mistral

    client = Mistral(api_key=mistral_key, cache=True, cache_path=tmp_path / "c.sqlite")
    questions = [f"Reply with the number {i} only." for i in range(5)]
    for question in questions * 2:  # 10 calls, 5 duplicates
        response = client.chat.complete(
            model=MISTRAL_MODEL, messages=user(question), max_tokens=MAX_TOKENS
        )
        assert response.choices
    again = asyncio.run(
        client.chat.complete_async(
            model=MISTRAL_MODEL, messages=user(questions[0]), max_tokens=MAX_TOKENS
        )
    )
    s = client.metrics.summary()
    record(
        "Mistral exact cache (11 calls, sync + async)",
        f"{s['cache_hits']} hits, ${s['cost_saved_usd']:.6f} saved, "
        f"cost ${s['cost_usd']:.6f}, p95 {s['latency_p95_ms']:.0f} ms",
    )
    assert caelum.cache_hit(again)
    assert s["cache_hits"] == 6
    assert s["api_calls"] == 5


OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Cheap models from different families, all served through an OpenAI-compatible API.
OPENROUTER_MODELS = [
    "mistralai/mistral-small-24b-instruct-2501",
    "deepseek/deepseek-v4-flash",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3.7-flash",
    "google/gemma-3-4b-it",
    "x-ai/grok-4.3",
    "microsoft/phi-4",
    "cohere/command-r7b-12-2024",
    "amazon/nova-micro-v1",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning",
    "ibm-granite/granite-4.2-8b",
    "minimax/minimax-m3.1-flash",
    "upstage/solar-mini4",
    "perplexity/sonar",
]


@pytest.mark.parametrize("model", OPENROUTER_MODELS)
def test_openai_compatible_provider_cache(model: str, openrouter_key: str, tmp_path: Path) -> None:
    # retry=True: some upstream providers rate-limit bursts (429); CAELUM retries them.
    client = OpenAI(
        base_url=OPENROUTER_URL,
        api_key=openrouter_key,
        cache=True,
        cache_path=tmp_path / "c.db",
        retry=True,
    )
    questions = [f"Reply with the number {i} only." for i in range(3)]
    for question in questions * 2:  # 6 calls, 3 duplicates
        completion = client.chat.completions.create(
            model=model, messages=user(question), max_tokens=32
        )
        assert completion.choices
    s = client.metrics.summary()
    record(
        f"OpenAI-compatible: {model} (6 calls)",
        f"{s['cache_hits']} hits, {s['retries']} retries, p95 {s['latency_p95_ms']:.0f} ms",
    )
    assert s["cache_hits"] == 3
    assert s["api_calls"] == 3


def test_openai_compatible_provider_streaming(openrouter_key: str) -> None:
    client = OpenAI(base_url=OPENROUTER_URL, api_key=openrouter_key)
    stream = client.chat.completions.create(
        model=OPENROUTER_MODELS[0], messages=user("Count from 1 to 5."), max_tokens=32, stream=True
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    record("OpenAI-compatible streaming (Mistral via OpenRouter)", f"{len(text)} characters")
    assert text
