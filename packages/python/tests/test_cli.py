"""`caelum cache stats`: what the cache file holds, without modifying it."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from caelum import OpenAI
from caelum.cache.stores import SQLiteStore
from caelum.cache.vectors import SQLiteVectorStore
from caelum.cli import cache_stats, format_stats, main
from tests.conftest import FakeServer


def openai_answer(model: str, prompt: int, completion: int) -> dict[str, Any]:
    return {"model": model, "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def fill(path: Path) -> None:
    store = SQLiteStore(path)
    store.set("a", openai_answer("gpt-4o-mini", 1000, 500), ttl=None)
    store.set("b", openai_answer("gpt-4o-mini-2024-07-18", 1000, 500), ttl=None)
    store.set(
        "c",
        {"model": "claude-haiku-5-5", "usage": {"input_tokens": 100, "output_tokens": 10}},
        ttl=None,
    )
    store.set("d", openai_answer("my-own-model", 10, 10), ttl=None)
    store.set("old", openai_answer("gpt-4o", 1, 1), ttl=0.001)
    store.close()
    vectors = SQLiteVectorStore(path)
    vectors.add(
        "ctx", "emb", [1.0, 0.0], openai_answer("gpt-4o-mini", 2000, 0), ttl=None, query="q"
    )
    vectors.close()


def test_stats_count_value_and_expiry(cache_path: Path) -> None:
    fill(cache_path)
    stats = cache_stats(cache_path, now=time.time() + 1)
    assert stats.exact_entries == 5
    assert stats.exact_expired == 1
    assert stats.semantic_entries == 1
    assert stats.live_entries == 5
    assert stats.size_bytes > 0
    mini = (1000 * 0.15 + 500 * 0.60) / 1e6
    haiku = (100 * 0.10 + 10 * 0.50) / 1e6
    assert stats.models["gpt-4o-mini"].entries == 2
    assert stats.models["gpt-4o-mini-2024-07-18"].entries == 1
    assert stats.value_usd == pytest.approx(2 * mini + 2000 * 0.15 / 1e6 + haiku)
    assert stats.unpriced_entries == 1
    assert "gpt-4o" not in stats.models  # expired entries are not valued


def test_report_text(cache_path: Path) -> None:
    fill(cache_path)
    text = format_stats(cache_stats(cache_path))
    assert "Exact entries:     5" in text
    assert "Semantic entries:  1" in text
    assert "my-own-model" in text
    assert "price unknown" in text


def test_stats_of_a_real_client_cache(server: FakeServer, cache_path: Path) -> None:
    client = OpenAI(
        api_key="sk-test", http_client=server.http_client(), cache=True, cache_path=cache_path
    )
    for i in (1, 2, 1):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"question {i}"}]
        )
    state: Any = getattr(client, "_caelum_state")  # noqa: B009
    state.exact_cache.store.flush()  # cache writes run in the background
    stats = cache_stats(cache_path)
    assert stats.exact_entries == 2
    assert stats.unpriced_entries == 0
    assert stats.value_usd > 0


def test_stats_do_not_modify_the_file(cache_path: Path) -> None:
    fill(cache_path)
    before = cache_path.read_bytes()
    cache_stats(cache_path, now=time.time() + 10)
    assert cache_path.read_bytes() == before


def test_cli_text_and_json(cache_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fill(cache_path)
    assert main(["cache", "stats", "--path", str(cache_path)]) == 0
    assert "CAELUM cache" in capsys.readouterr().out
    assert main(["cache", "stats", "--path", str(cache_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["exact_entries"] == 5
    assert data["models"]["claude-haiku-5-5"]["entries"] == 1


def test_cli_default_path_uses_caelum_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAELUM_HOME", str(tmp_path))
    fill(tmp_path / "cache.sqlite")
    assert main(["cache", "stats"]) == 0
    assert str(tmp_path) in capsys.readouterr().out


def test_cli_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cache", "stats", "--path", str(tmp_path / "nope.sqlite")]) == 1
    assert "no cache file" in capsys.readouterr().err


def test_cli_rejects_other_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junk = tmp_path / "junk.sqlite"
    junk.write_text("not a database", encoding="utf-8")
    assert main(["cache", "stats", "--path", str(junk)]) == 1
    assert "not a" in capsys.readouterr().err
    import sqlite3

    other = tmp_path / "other.sqlite"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE t (x)")
    conn.close()
    assert main(["cache", "stats", "--path", str(other)]) == 1
    assert "not a CAELUM cache" in capsys.readouterr().err


def test_python_dash_m_caelum(cache_path: Path) -> None:
    fill(cache_path)
    result = subprocess.run(
        [sys.executable, "-m", "caelum", "cache", "stats", "--path", str(cache_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["semantic_entries"] == 1


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.startswith("caelum 0.")
