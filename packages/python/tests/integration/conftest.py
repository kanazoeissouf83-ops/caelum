"""Integration tests against the REAL provider APIs (they cost money).

Excluded by default. Run them explicitly:

    cp .env.example .env      # then fill in your keys
    uv run pytest -m integration

Cost guard: cheapest models, ``max_tokens=16``, about 100 short requests in total —
well under 1 EUR per full run. Tests needing a missing key are skipped.

Models can be overridden with ``CAELUM_IT_OPENAI_MODEL`` / ``CAELUM_IT_ANTHROPIC_MODEL``.
A summary of the measurements is written to ``benchmarks/results/<date>-integration.md``.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
MAX_TOKENS = 16


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency). Existing variables win."""
    path = PACKAGE_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

OPENAI_MODEL = os.environ.get("CAELUM_IT_OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("CAELUM_IT_ANTHROPIC_MODEL", "claude-haiku-5-5")
GEMINI_MODEL = os.environ.get("CAELUM_IT_GEMINI_MODEL", "gemini-3.1-flash-lite")
MISTRAL_MODEL = os.environ.get("CAELUM_IT_MISTRAL_MODEL", "mistral-small-latest")

_REPORT: list[tuple[str, str]] = []


def record(test: str, result: str) -> None:
    """Add a line to the integration report."""
    _REPORT.append((test, result))


@pytest.fixture
def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key


@pytest.fixture
def mistral_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


@pytest.fixture
def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture(scope="session", autouse=True)
def _write_report() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    yield
    if not _REPORT:
        return
    today = dt.date.today().isoformat()
    target = REPO_ROOT / "benchmarks" / "results" / f"{today}-integration.md"
    # WHY: running a subset (e.g. `-k gemini`) must update its own rows only, not erase
    # the results of the other tests run earlier the same day.
    rows: dict[str, str] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if line.startswith("| ") and len(cells) == 2 and cells[0] not in ("Test", "---"):
                rows[cells[0]] = cells[1]
    rows.update(_REPORT)
    lines = [
        f"# Integration tests — real APIs ({today})",
        "",
        f"Models: OpenAI `{OPENAI_MODEL}`, Anthropic `{ANTHROPIC_MODEL}`, "
        f"Gemini `{GEMINI_MODEL}`, Mistral `{MISTRAL_MODEL}` — `max_tokens={MAX_TOKENS}`.",
        "",
        "| Test | Result |",
        "| --- | --- |",
        *(f"| {test} | {result} |" for test, result in rows.items()),
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def user(text: str) -> list[Any]:
    return [{"role": "user", "content": text}]
