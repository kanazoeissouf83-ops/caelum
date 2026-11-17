"""Task 3: lifetime of cached answers per end customer, and clearing a customer's cache."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import OpenAI
from caelum import _engine as engine_module
from caelum.cache import SQLiteStore, SQLiteVectorStore
from caelum.cache.ttl import DEFAULT_TTL, MAX_TTL, TtlPolicy, check_ttl
from caelum.cli import main as cli_main
from caelum.cloud import customer_tag
from tests.conftest import FakeServer
from tests.test_cloud import KEY, FakeCloud, connection_of

HOUR = 3600.0
DAY = 86_400.0


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCloud]:
    """A fake CAELUM Cloud (same as in test_cloud.py)."""
    fake = FakeCloud()
    monkeypatch.setenv("CAELUM_CLOUD_URL", fake.url)
    monkeypatch.delenv("CAELUM_API_KEY", raising=False)
    engine_module._cloud_connections.clear()  # pyright: ignore[reportPrivateUsage]
    yield fake
    for connection in engine_module._cloud_connections.values():  # pyright: ignore[reportPrivateUsage]
        connection.close()
    engine_module._cloud_connections.clear()  # pyright: ignore[reportPrivateUsage]
    fake.close()


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def client(server: FakeServer, tmp_path: Path, store: Any = None, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {"cache": True, "cache_admission": 1, **kwargs}
    if store is not None:
        options["cache_store"] = store
    else:
        options.setdefault("cache_path", tmp_path / "ttl.sqlite")
    options.setdefault("journal_path", tmp_path / "journal.sqlite")
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, question: str = "Quels sont vos horaires ?") -> Any:
    return c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )


# -- 3.1 settings ----------------------------------------------------------------------


def test_the_default_lifetime_is_24_hours(server: FakeServer, tmp_path: Path) -> None:
    clock = Clock()
    c = client(server, tmp_path, SQLiteStore(tmp_path / "a.sqlite", clock=clock))
    assert c.__dict__["_caelum_state"].ttl.default == DEFAULT_TTL == 24 * HOUR
    ask(c)
    clock.now += 24 * HOUR - 60
    assert caelum.cache_hit(ask(c))
    clock.now += 120
    assert not caelum.cache_hit(ask(c))


def test_the_lifetime_can_be_set_per_customer(server: FakeServer, tmp_path: Path) -> None:
    clock = Clock()
    c = client(
        server, tmp_path, SQLiteStore(tmp_path / "b.sqlite", clock=clock),
        cache_customer_ttl={"promo-shop": HOUR},
    )  # fmt: skip
    with caelum.customer("promo-shop"):
        ask(c)
    with caelum.customer("other"):
        ask(c)
    clock.now += 2 * HOUR
    with caelum.customer("promo-shop"):
        assert not caelum.cache_hit(ask(c))  # 1 hour for this customer
    with caelum.customer("other"):
        assert caelum.cache_hit(ask(c))  # 24 hours (default) for the others


@pytest.mark.parametrize("bad", [0, -5, "1h", 30, 31 * DAY, True, float("nan")])
def test_invalid_lifetimes_are_refused_with_a_clear_french_message(
    server: FakeServer, tmp_path: Path, bad: Any
) -> None:
    with pytest.raises(ValueError, match=r"cache_ttl.*(durée|secondes)"):
        client(server, tmp_path, cache_ttl=bad)
    with pytest.raises(ValueError, match=r"cache_customer_ttl\['acme'\].*(durée|secondes)"):
        client(server, tmp_path, cache_customer_ttl={"acme": bad})
    assert not (tmp_path / "ttl.sqlite").exists()  # refused before any cache file is opened


def test_none_means_the_30_day_maximum() -> None:
    assert check_ttl(None) == MAX_TTL == 30 * DAY
    assert TtlPolicy(None).default == MAX_TTL


# -- 3.2 shortening applies to stored entries, lengthening does not ---------------------


def test_a_shorter_lifetime_applies_to_answers_already_cached(
    server: FakeServer, tmp_path: Path
) -> None:
    clock = Clock()
    c = client(server, tmp_path, SQLiteStore(tmp_path / "c.sqlite", clock=clock))
    with caelum.customer("acme"):
        ask(c)  # cached for 24 hours
    clock.now += 2 * HOUR
    c.__dict__["_caelum_state"].ttl.set_customer("acme", HOUR)  # shortened to 1 hour
    with caelum.customer("acme"):
        assert not caelum.cache_hit(ask(c))  # 2 hours old > 1 hour: no longer served


def test_a_longer_lifetime_does_not_extend_answers_already_cached(
    server: FakeServer, tmp_path: Path
) -> None:
    clock = Clock()
    c = client(
        server, tmp_path, SQLiteStore(tmp_path / "d.sqlite", clock=clock),
        cache_customer_ttl={"acme": HOUR},
    )  # fmt: skip
    with caelum.customer("acme"):
        ask(c)  # cached for 1 hour
    c.__dict__["_caelum_state"].ttl.set_customer("acme", 10 * DAY)
    clock.now += 2 * HOUR
    with caelum.customer("acme"):
        assert not caelum.cache_hit(ask(c))  # it kept its original 1-hour expiry


def test_the_semantic_cache_honours_a_shortened_lifetime(tmp_path: Path) -> None:
    clock = Clock()
    store = SQLiteVectorStore(tmp_path / "v.sqlite", clock=clock)
    store.add("ctx", "m", [1.0, 0.0], {"a": 1}, DAY, query="q", customer="acme")
    clock.now += 2 * HOUR
    assert store.search("ctx", "m", [1.0, 0.0], 0.9) is not None
    assert store.search("ctx", "m", [1.0, 0.0], 0.9, max_age=HOUR) is None
    store.close()


# -- 3.2 CAELUM Cloud ----------------------------------------------------------------------


def test_cloud_entries_use_the_same_expiry_rule(
    server: FakeServer,
    cloud: FakeCloud,
    tmp_path: Path,
) -> None:
    c = client(server, tmp_path, caelum_key=KEY, cache_customer_ttl={"acme": HOUR})
    with caelum.customer("acme"):
        ask(c)
    connection_of(c).flush()
    (name,) = cloud.cache
    assert name.startswith(customer_tag("acme") + ":")  # filed under the customer's hash
    assert "acme" not in name  # never the customer's name
    envelope = json.loads(cloud.cache[name])
    assert envelope["ttl"] == HOUR and cloud.ttls[name] == HOUR  # same lifetime sent
    assert abs(envelope["created"] - cloud.created[name]) < 5  # same start: now


def test_an_expired_cloud_entry_is_not_served(
    server: FakeServer,
    cloud: FakeCloud,
    tmp_path: Path,
) -> None:
    first = client(server, tmp_path, caelum_key=KEY, cache_path=tmp_path / "s1.sqlite")
    with caelum.customer("acme"):
        ask(first)
    connection_of(first).flush()
    (name,) = cloud.cache
    envelope = json.loads(cloud.cache[name])
    envelope["created"] = time.time() - 2 * HOUR  # written two hours ago
    cloud.cache[name] = json.dumps(envelope)
    second = client(
        server, tmp_path, caelum_key=KEY, cache_path=tmp_path / "s2.sqlite",
        cache_customer_ttl={"acme": HOUR},
    )  # fmt: skip
    with caelum.customer("acme"):
        assert not caelum.cache_hit(ask(second))  # older than this customer's 1 hour


# -- 3.3 clearing -------------------------------------------------------------------------


def test_clearing_a_customer_deletes_local_and_cloud_entries_only_for_it(
    server: FakeServer,
    cloud: FakeCloud,
    tmp_path: Path,
) -> None:
    c = client(server, tmp_path, caelum_key=KEY)
    for customer in ("boutique-a", "boutique-b"):
        with caelum.customer(customer):
            ask(c, "Quels sont vos horaires ?")
            ask(c, "Livrez-vous à Bobo-Dioulasso ?")
    connection_of(c).flush()
    assert len(cloud.cache) == 4
    assert c.clear_cache("boutique-a") == {"local": 2, "cloud": 2}
    assert len(cloud.cache) == 2
    with caelum.customer("boutique-a"):
        assert not caelum.cache_hit(ask(c, "Quels sont vos horaires ?"))
    with caelum.customer("boutique-b"):
        assert caelum.cache_hit(ask(c, "Quels sont vos horaires ?"))


def test_clearing_only_old_entries(server: FakeServer, tmp_path: Path) -> None:
    clock = Clock()
    store = SQLiteStore(tmp_path / "e.sqlite", clock=clock)
    c = client(server, tmp_path, store, cache_ttl=30 * DAY)  # nothing expires on its own
    with caelum.customer("acme"):
        ask(c, "Quels sont vos horaires ?")
        clock.now += 3 * DAY
        ask(c, "Livrez-vous à Bobo-Dioulasso ?")
        assert c.clear_cache("acme", older_than=2 * DAY)["local"] == 1
        assert not caelum.cache_hit(ask(c, "Quels sont vos horaires ?"))  # deleted
        assert caelum.cache_hit(ask(c, "Livrez-vous à Bobo-Dioulasso ?"))  # recent: kept


def test_every_clearing_is_noted_in_the_journal(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    with caelum.customer("acme"):
        ask(c)
    c.clear_cache("acme")
    c.clear_cache("acme", older_than=7 * DAY)
    journal = c.journal
    assert journal is not None
    purges = [r for r in journal.entries() if r["outcome"] == "purge"]
    assert [(r["customer"], r["reason"]) for r in purges] == [
        ("acme", "clear"),
        ("acme", "older_than:604800s"),
    ]
    assert purges[0]["entry_id"] == "local:1 cloud:-"
    assert journal.stats("acme")["requests"] == 1  # a clearing is not a request


def test_invalid_clearing_requests_are_refused(server: FakeServer, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    with pytest.raises(ValueError, match="older_than"):
        c.clear_cache("acme", older_than=0)
    with pytest.raises(TypeError):
        c.clear_cache(None)  # type: ignore[arg-type]


def test_the_command_line_clears_a_customer(
    server: FakeServer, tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CAELUM_API_KEY", raising=False)
    c = client(server, tmp_path)
    for customer in ("acme", "other"):
        with caelum.customer(customer):
            ask(c)
    state = c.__dict__["_caelum_state"]
    state.exact_cache.store.flush()
    path = str(tmp_path / "ttl.sqlite")
    assert cli_main(["cache", "clear", "--customer", "acme", "--path", path]) == 0
    assert "1 cached answer(s) of customer 'acme' deleted locally" in capsys.readouterr().out
    with caelum.customer("other"):
        assert caelum.cache_hit(ask(c))
