"""Improvement B: frequency admission counted for the whole account in CAELUM Cloud."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import caelum
from caelum import OpenAI
from caelum import _engine as engine_module
from tests.conftest import FakeServer
from tests.test_cloud import KEY, FakeCloud, connection_of

QUESTION = "Quels sont vos horaires ?"


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCloud]:
    fake = FakeCloud()
    monkeypatch.setenv("CAELUM_CLOUD_URL", fake.url)
    monkeypatch.delenv("CAELUM_API_KEY", raising=False)
    engine_module._cloud_connections.clear()  # pyright: ignore[reportPrivateUsage]
    yield fake
    for connection in engine_module._cloud_connections.values():  # pyright: ignore[reportPrivateUsage]
        connection.close()
    engine_module._cloud_connections.clear()  # pyright: ignore[reportPrivateUsage]
    fake.close()


def server_app(server: FakeServer, tmp_path: Path, name: str, **kwargs: Any) -> OpenAI:
    """One app server: its own local cache file, the same CAELUM key."""
    options: dict[str, Any] = {
        "caelum_key": KEY,
        "cache": True,
        "cache_admission": 2,
        "cache_path": tmp_path / f"{name}.sqlite",
        "journal_path": tmp_path / f"{name}-journal.sqlite",
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: Any, question: str = QUESTION) -> Any:
    return c.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )


def settle(*apps: OpenAI) -> None:
    """Let the background uploads (counts, then admitted answers) reach the Cloud."""
    for _ in range(2):
        for app in apps:
            connection_of(app).flush()


def test_once_on_each_of_two_servers_is_admitted(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    one, two, three = (server_app(server, tmp_path, f"s{i}") for i in (1, 2, 3))
    ask(one)
    settle(one)
    ask(two)
    settle(two)
    assert caelum.cache_hit(ask(three))
    assert server.calls == 2
    assert two.metrics.summary()["admission_shared_admitted"] == 1
    assert one.journal is not None
    assert one.journal.entries()[0]["admission_counter"] == "shared"


def test_three_servers_with_n_3(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    apps = [server_app(server, tmp_path, f"t{i}", cache_admission=3) for i in range(4)]
    for app in apps[:3]:
        assert not caelum.cache_hit(ask(app))
        settle(app)
    assert caelum.cache_hit(ask(apps[3]))
    assert server.calls == 3


def test_two_servers_counting_at_the_same_time(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    one, two = server_app(server, tmp_path, "c1"), server_app(server, tmp_path, "c2")
    threads = [threading.Thread(target=ask, args=(app,)) for app in (one, two)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    flushes = [threading.Thread(target=settle, args=(app,)) for app in (one, two)]
    for thread in flushes:
        thread.start()
    for thread in flushes:
        thread.join()
    settle(one, two)
    admitted = sum(app.metrics.summary()["admission_shared_admitted"] for app in (one, two))
    assert admitted == 1  # both counts kept, exactly one admission
    assert cloud.candidates == {}


def test_an_unreachable_cloud_falls_back_to_the_local_counter(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    cloud.mode = "down"
    app = server_app(server, tmp_path, "d1")
    assert not caelum.cache_hit(ask(app))
    settle(app)
    assert not caelum.cache_hit(ask(app))  # 2nd answer on this server: admitted locally
    assert caelum.cache_hit(ask(app))
    assert app.metrics.summary()["admission_shared_unavailable"] >= 1


def test_an_excluded_question_is_never_sent_to_the_cloud(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    one, two = server_app(server, tmp_path, "e1"), server_app(server, tmp_path, "e2")
    for app in (one, two):
        ask(app, "Rappelez-moi au +226 70 12 34 56")
        settle(app)
    assert cloud.admission_bodies == []
    assert all("70 12 34 56" not in path for _, path in cloud.requests)


def test_only_hashes_are_sent(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    app = server_app(server, tmp_path, "h1", cache_customer_ttl={"boutique": 3600})
    with caelum.customer("boutique"):
        ask(app)
    settle(app)
    (body,) = cloud.admission_bodies
    assert set(body) == {"fingerprint", "n", "ttl"}
    assert re.fullmatch(r"[0-9a-f]{64}", body["fingerprint"])
    assert (body["n"], body["ttl"]) == (2, 3600)  # the customer's own lifetime
    assert "horaires" not in json.dumps(cloud.admission_bodies)
    assert all("boutique" not in path for _, path in cloud.requests)


def test_end_customers_are_counted_apart(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    one, two, three = (server_app(server, tmp_path, f"i{i}") for i in (1, 2, 3))
    with caelum.customer("boutique-a"):
        ask(one)
    settle(one)
    with caelum.customer("boutique-b"):
        ask(two)
    settle(two)
    with caelum.customer("boutique-b"):
        assert not caelum.cache_hit(ask(three))  # A's count never helps B


def test_pending_shared_candidates(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    app = server_app(server, tmp_path, "p1")
    with caelum.customer("acme"):
        ask(app)
    settle(app)
    assert app.shared_admission_pending("acme") == 1
    assert app.shared_admission_pending("other") == 0
    no_cloud = OpenAI(api_key="sk-test", http_client=server.http_client(), cache=True)
    assert no_cloud.shared_admission_pending() is None


def test_without_cloud_the_counter_stays_local(server: FakeServer, tmp_path: Path) -> None:
    app = OpenAI(
        api_key="sk-test",
        http_client=server.http_client(),
        cache=True,
        cache_admission=2,
        cache_path=tmp_path / "local.sqlite",
        journal_path=tmp_path / "local-journal.sqlite",
    )
    ask(app)
    assert app.journal is not None
    assert app.journal.entries()[0]["admission_counter"] == "local"


def test_a_free_account_keeps_the_local_counter_and_stops_asking(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    cloud.mode = "free_plan"  # the shared counter is a Starter feature
    app = server_app(server, tmp_path, "f1")
    ask(app)
    settle(app)
    ask(app, "Livrez-vous à Bobo-Dioulasso ?")
    settle(app)
    admission_calls = [r for r in cloud.requests if r[1].startswith("/v1/admission")]
    assert len(admission_calls) == 1  # refused once, then not asked again
    assert not caelum.cache_hit(ask(app))  # 2nd answer on this server
    assert caelum.cache_hit(ask(app))  # admitted by the local counter
