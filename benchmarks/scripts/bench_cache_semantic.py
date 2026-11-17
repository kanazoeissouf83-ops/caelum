"""Calibrate the semantic cache threshold on real embeddings.

Embeds the calibration set (``packages/python/tests/fixtures/semantic_pairs.json``:
paraphrase pairs that SHOULD share an answer, and look-alike pairs that must NOT), then
measures, for every threshold from 0.80 to 0.99:

* reuse rate: share of paraphrase pairs that would be served from the cache;
* false-positive rate: share of look-alike pairs that would wrongly reuse an answer.

The recommended default is the **lowest threshold with zero false positives** on this
set (zero false positives is the priority; 0.97 is only the starting point).

Cost: one embeddings request for ~200 short texts with text-embedding-3-small, i.e. a
tiny fraction of a cent.

Run from the repository root:

    export OPENAI_API_KEY=sk-...
    uv run --project packages/python python benchmarks/scripts/bench_cache_semantic.py \\
        --output benchmarks/results/$(date +%F)-cache-semantic.md

``--offline`` uses a deterministic fake embedder to check the script itself (results
are meaningless and clearly flagged as such).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from caelum.cache.semantic import DEFAULT_EMBEDDING_MODEL, DEFAULT_SEMANTIC_THRESHOLD

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "packages" / "python" / "tests" / "fixtures" / "semantic_pairs.json"
THRESHOLDS = [round(0.80 + i / 100, 2) for i in range(20)]


def load_pairs() -> tuple[list[list[str]], list[list[str]]]:
    data: dict[str, Any] = json.loads(DATASET.read_text(encoding="utf-8"))
    return data["similar"], data["different"]


def embed_openai(texts: list[str], model: str) -> dict[str, list[float]]:
    import openai

    client = openai.OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    return {text: list(item.embedding) for text, item in zip(texts, response.data, strict=True)}


def embed_offline(texts: list[str]) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    for text in texts:
        digest = hashlib.sha256(text.lower().encode()).digest()
        vectors[text] = [(b / 255) * 2 - 1 for b in digest]
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b)))


def run(offline: bool, model: str) -> dict[str, Any]:
    similar, different = load_pairs()
    texts = sorted({text for pair in [*similar, *different] for text in pair})
    vectors = embed_offline(texts) if offline else embed_openai(texts, model)
    similar_scores = [cosine(vectors[a], vectors[b]) for a, b in similar]
    different_scores = [cosine(vectors[a], vectors[b]) for a, b in different]
    curve: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        reuse = sum(score >= threshold for score in similar_scores) / len(similar_scores)
        false_pos = sum(score >= threshold for score in different_scores) / len(different_scores)
        curve.append({"threshold": threshold, "reuse": reuse, "false_positive": false_pos})
    safe = [point for point in curve if point["false_positive"] == 0]
    recommended = safe[0]["threshold"] if safe else None
    hardest_different = sorted(
        zip(different_scores, different, strict=True), key=lambda item: -item[0]
    )[:5]
    weakest_similar = sorted(zip(similar_scores, similar, strict=True), key=lambda item: item[0])[
        :5
    ]
    return {
        "offline": offline,
        "model": "offline-fake" if offline else model,
        "texts": len(texts),
        "similar": len(similar),
        "different": len(different),
        "curve": curve,
        "recommended": recommended,
        "hardest_different": hardest_different,
        "weakest_similar": weakest_similar,
    }


def render(results: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {p['threshold']:.2f} | {p['reuse']:.0%} | {p['false_positive']:.0%} |"
        for p in results["curve"]
    )
    hardest = "\n".join(
        f"| {score:.4f} | {a} | {b} |" for score, (a, b) in results["hardest_different"]
    )
    weakest = "\n".join(
        f"| {score:.4f} | {a} | {b} |" for score, (a, b) in results["weakest_similar"]
    )
    recommended = results["recommended"]
    verdict = (
        f"**Recommended default threshold: {recommended:.2f}** (lowest threshold with zero "
        "false positives on this set)."
        if recommended is not None
        else "**No threshold in range has zero false positives** — keep the strict default "
        "and enlarge the calibration set."
    )
    warning = (
        "\n> **OFFLINE RUN — fake embeddings, results are meaningless.** Re-run with an "
        "OpenAI API key.\n"
        if results["offline"]
        else ""
    )
    return f"""# Benchmark — semantic cache calibration ({dt.date.today().isoformat()})
{warning}
Embedding model: `{results["model"]}` — {results["texts"]} texts,
{results["similar"]} paraphrase pairs, {results["different"]} look-alike pairs.
Current default: {DEFAULT_SEMANTIC_THRESHOLD}.

{verdict}

## Reuse / false-positive curve

| Threshold | Reuse (paraphrases) | False positives (look-alikes) |
|---|---|---|
{rows}

## Most dangerous look-alike pairs (highest similarity)

| Cosine | Question A | Question B |
|---|---|---|
{hardest}

## Hardest paraphrases (lowest similarity)

| Cosine | Question A | Question B |
|---|---|---|
{weakest}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the markdown report to this file")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="embedding model")
    parser.add_argument("--offline", action="store_true", help="fake embeddings (script check)")
    args = parser.parse_args()
    if not args.offline and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (or use --offline to test the script).", file=sys.stderr)
        raise SystemExit(2)
    report = render(run(args.offline, args.model))
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
