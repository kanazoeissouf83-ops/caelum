"""Task 5: false-positive rate of the semantic cache on 138 recorded question pairs.

The calibration pairs (``fixtures/semantic_pairs.json``: 69 paraphrases and 69 look-alikes,
English, French and West African) were embedded with ``text-embedding-3-small`` and judged
by ``gpt-4o-mini`` once (``benchmarks/scripts/record_semantic_pairs.py``). This test
replays them offline through the real CAELUM pipeline with the default settings:

* **embeddings only** (``semantic_verifier=False``, threshold 0.97, gray zone from 0.92);
* **with the judge** (the default with OpenAI: threshold 0.60, both orders must say YES).

For each pair the first question is asked, then the second. A look-alike served from the
cache is a **false positive**; a paraphrase served from the cache is a **reuse**. The rates
are printed at the end of the test run, and any false positive fails the suite.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import OpenAI
from caelum.cache import EmbeddingResult
from caelum.cache.verifier import Verdict
from tests.conftest import FakeServer

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED = json.loads((FIXTURES / "semantic_pairs_recorded.json").read_text(encoding="utf-8"))
DATASET = json.loads((FIXTURES / "semantic_pairs.json").read_text(encoding="utf-8"))


def unpack(encoded: str) -> list[float]:
    raw = base64.b64decode(encoded)
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


VECTORS = {question: unpack(encoded) for question, encoded in RECORDED["vectors"].items()}
VERDICTS = {(p["stored"], p["new"]): tuple(p["judge"]) for p in RECORDED["pairs"]}


class RecordedEmbedder:
    @property
    def model(self) -> str:
        return str(RECORDED["embedding_model"])

    def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=VECTORS[text], tokens=1)


class RecordedJudge:
    """Replays the verdicts recorded for (stored, new), in both orders."""

    model = str(RECORDED["judge_model"])

    def verify(self, stored: str, new: str) -> Verdict:
        if (stored, new) in VERDICTS:
            forward, backward = VERDICTS[(stored, new)]
        else:
            backward, forward = VERDICTS[(new, stored)]
        return Verdict(accepted=bool(forward and backward))


@dataclass
class Rates:
    mode: str
    paraphrases: int = 0
    reused: int = 0
    lookalikes: int = 0
    false_positives: list[tuple[str, str]] = field(default_factory=lambda: [])
    excluded: int = 0  # pairs kept out of the cache by the automatic exclusions
    fr_paraphrases: int = 0  # French / West African pairs (after the first 50 of each kind)
    fr_reused: int = 0
    fr_lookalikes: int = 0
    fr_false_positives: int = 0

    def line(self) -> str:
        fp = len(self.false_positives)
        return (
            f"{self.mode:<26} false positives {fp}/{self.lookalikes} "
            f"({100 * fp / max(self.lookalikes, 1):.1f} %)   "
            f"paraphrases reused {self.reused}/{self.paraphrases} "
            f"({100 * self.reused / max(self.paraphrases, 1):.1f} %)   "
            f"excluded pairs {self.excluded}\n"
            f"{'  of which French':<26} false positives {self.fr_false_positives}/"
            f"{self.fr_lookalikes}   paraphrases reused {self.fr_reused}/{self.fr_paraphrases}"
        )


RESULTS: list[Rates] = []  # printed by conftest.pytest_terminal_summary


def replay(server: FakeServer, tmp_path: Path, mode: str, **options: Any) -> Rates:
    rates = Rates(mode)
    pairs = [("similar", i, a, b) for i, (a, b) in enumerate(DATASET["similar"])] + [
        ("different", i, a, b) for i, (a, b) in enumerate(DATASET["different"])
    ]
    for index, (kind, position, stored, new) in enumerate(pairs):
        french = position >= 50
        # A fresh cache per pair: only this pair's first question can be matched.
        client = OpenAI(
            api_key="sk-test",
            http_client=server.http_client(),
            semantic_cache=True,
            semantic_embedder=RecordedEmbedder(),
            cache_path=tmp_path / f"{mode}-{index}.sqlite",
            cache_admission=1,
            journal=False,
            **options,
        )
        for question in (stored, new):
            response: Any = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
            )
        excluded = client.metrics.summary()["excluded_requests"] > 0
        hit = caelum.cache_type(response) == "semantic"
        if excluded:
            rates.excluded += 1
        elif kind == "similar":
            rates.paraphrases += 1
            rates.reused += hit
            rates.fr_paraphrases += french
            rates.fr_reused += french and hit
        else:
            rates.lookalikes += 1
            rates.fr_lookalikes += french
            if hit:
                rates.false_positives.append((stored, new))
                rates.fr_false_positives += french
    RESULTS.append(rates)
    return rates


def test_the_calibration_set_is_large_enough_and_recorded() -> None:
    assert len(DATASET["similar"]) >= 30 and len(DATASET["different"]) >= 30
    recorded = {(p["stored"], p["new"]) for p in RECORDED["pairs"]}
    expected = {tuple(p) for p in DATASET["similar"] + DATASET["different"]}
    assert recorded == expected, "pairs changed: run benchmarks/scripts/record_semantic_pairs.py"


def test_no_false_positive_with_embeddings_only(server: FakeServer, tmp_path: Path) -> None:
    rates = replay(server, tmp_path, "embeddings only (0.97)", semantic_verifier=False)
    assert rates.false_positives == [], rates.line()


def test_no_false_positive_with_the_judge(server: FakeServer, tmp_path: Path) -> None:
    rates = replay(server, tmp_path, "with the judge (0.60)", semantic_verifier=RecordedJudge())
    assert rates.false_positives == [], rates.line()
    assert rates.reused > 0  # the judge lets real paraphrases through


@pytest.mark.parametrize("threshold", [0.90, 0.85])
def test_lower_thresholds_without_a_judge_are_measured(
    server: FakeServer, tmp_path: Path, threshold: float
) -> None:
    # Documents why 0.97 is the default without a judge: below it, look-alikes get through.
    replay(
        server,
        tmp_path,
        f"embeddings only ({threshold:.2f})",
        semantic_verifier=False,
        semantic_threshold=threshold,
        semantic_gray_threshold=threshold,
    )
