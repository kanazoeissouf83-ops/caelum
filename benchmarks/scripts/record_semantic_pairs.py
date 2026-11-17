"""Record real embeddings and judge verdicts for the calibration pairs (real API calls).

Reads ``packages/python/tests/fixtures/semantic_pairs.json`` and writes
``semantic_pairs_recorded.json`` next to it: the ``text-embedding-3-small`` vector of every
question (float16, base64) and the ``gpt-4o-mini`` judge verdict in both orders for every
pair. The test suite replays them offline (``tests/test_semantic_pairs.py``), so that the
false-positive rate of the semantic cache is measured on every run without calling any
API.

Run again after changing the pairs, the judge prompt or the models:

    cd packages/python && uv run python ../../benchmarks/scripts/record_semantic_pairs.py

Cost: about $0.02 (280 embeddings and at most 280 short judge calls).
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "packages" / "python" / "tests" / "fixtures"
DATASET = FIXTURES / "semantic_pairs.json"
OUTPUT = FIXTURES / "semantic_pairs_recorded.json"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def pack(vector: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}e", *vector)).decode("ascii")


def main() -> int:
    load_env(ROOT / "packages" / "python" / ".env")
    from caelum import LLMVerifier, OpenAIEmbedder

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    pairs = [("similar", a, b) for a, b in data["similar"]] + [
        ("different", a, b) for a, b in data["different"]
    ]
    embedder = OpenAIEmbedder()
    judge = LLMVerifier(symmetric=False)  # one order per call: both are recorded
    questions = sorted({q for _, a, b in pairs for q in (a, b)})
    vectors: dict[str, str] = {}
    for i, question in enumerate(questions, 1):
        vectors[question] = pack(embedder.embed(question).vector)
        print(f"embedding {i}/{len(questions)}", file=sys.stderr)
    verdicts: list[dict[str, object]] = []
    for i, (kind, stored, new) in enumerate(pairs, 1):
        forward = judge.verify(stored, new).accepted
        backward = judge.verify(new, stored).accepted
        verdicts.append(
            {"kind": kind, "stored": stored, "new": new, "judge": [forward, backward]}
        )
        print(f"judge {i}/{len(pairs)}", file=sys.stderr)
    OUTPUT.write_text(
        json.dumps(
            {
                "recorded": dt.date.today().isoformat(),
                "embedding_model": embedder.model,
                "judge_model": judge.model,
                "vectors": vectors,
                "pairs": verdicts,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUTPUT} ({len(questions)} questions, {len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
