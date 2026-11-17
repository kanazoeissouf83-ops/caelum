"""CAELUM Cloud connection: metrics, shared cache, and never breaking the app."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import openai
import pytest
from openai.types.chat import ChatCompletion

import caelum
from caelum import AsyncOpenAI, OpenAI
from caelum import _engine as engine_module  # pyright: ignore[reportPrivateUsage]
from caelum.cloud import CloudConnection, current_customer, resolve_cloud_key
from tests.conftest import FakeServer

KEY = "sk-caelum-test-key"
MESSAGES: list[Any] = [{"role": "user", "content": "Where is my order?"}]


class FakeCloud:
    """Minimal CAELUM Cloud API: same routes and status codes as the real Worker."""

    def __init__(self) -> None:
        self.cache: dict[str, str] = {}  # "<customer tag>:<key>" -> envelope
        self.created: dict[str, float] = {}
        self.ttls: dict[str, float | None] = {}
        self.candidates: dict[tuple[str, str], int] = {}  # (tag, fingerprint) -> count
        self.admission_bodies: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str]] = []
        self.mode = "ok"  # ok | down | slow | slow_lookups | quota | revoked
        self.connections = 0  # TCP connections opened by the SDK
        cloud = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, like the real API

            def setup(self) -> None:
                cloud.connections += 1
                super().setup()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def _reply(self, status: int, body: bytes = b"") -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str) -> None:
                cloud.requests.append((method, self.path))
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                if cloud.mode == "slow" or (
                    cloud.mode == "slow_lookups" and method == "GET" and "/v1/cache/" in self.path
                ):
                    time.sleep(2)
                if cloud.mode == "down":
                    return self._reply(503)
                if cloud.mode == "revoked" or self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self._reply(401)
                if self.path == "/v1/events" and method == "POST":
                    cloud.events.extend(json.loads(body)["events"])
                    return self._reply(200, b"{}")
                if self.path.startswith("/v1/admission"):
                    if cloud.mode == "free_plan":
                        return self._reply(403, b'{"error":{"code":"plan_required"}}')
                    tag = self.path.partition("c=")[2] or "-"
                    if method == "GET":
                        pending = sum(1 for (t, _) in cloud.candidates if t == tag)
                        return self._reply(200, json.dumps({"pending": pending}).encode())
                    data = json.loads(body)
                    cloud.admission_bodies.append(data)
                    with cloud.lock:  # atomic, like the real D1 upsert
                        slot = (tag, data["fingerprint"])
                        count = cloud.candidates.get(slot, 0) + 1
                        cloud.candidates[slot] = count
                        if count >= data["n"]:
                            del cloud.candidates[slot]
                    reply = {"count": count, "admitted": count == data["n"]}
                    return self._reply(200, json.dumps(reply).encode())
                if self.path.startswith("/v1/cache"):
                    if cloud.mode == "quota":
                        return self._reply(429)
                    path, _, raw_query = self.path.partition("?")
                    query = dict(p.split("=", 1) for p in raw_query.split("&") if "=" in p)
                    tag = query.get("c", "-")
                    if method == "DELETE" and path == "/v1/cache":
                        older = float(query["older_than"]) if "older_than" in query else None
                        doomed = [
                            k
                            for k in cloud.cache
                            if k.startswith(f"{tag}:")
                            and (older is None or cloud.created[k] <= time.time() - older)
                        ]
                        for k in doomed:
                            del cloud.cache[k]
                        return self._reply(200, json.dumps({"deleted": len(doomed)}).encode())
                    key = f"{tag}:{path.split('/')[3]}"
                    if method == "PUT":
                        cloud.cache[key] = body.decode()
                        cloud.created[key] = time.time()
                        cloud.ttls[key] = float(query["ttl"]) if "ttl" in query else None
                        return self._reply(204)
                    if key in cloud.cache:
                        return self._reply(200, cloud.cache[key].encode())
                    return self._reply(404)
                return self._reply(404)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_PUT(self) -> None:
                self._handle("PUT")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


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


def connection_of(client: Any) -> CloudConnection:
    state = client.__dict__["_caelum_state"]
    assert state.cloud is not None
    return cast(CloudConnection, state.cloud)


def client(server: FakeServer, tmp_path: Path, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {"caelum_key": KEY, **kwargs}
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def ask(c: OpenAI, **kwargs: Any) -> ChatCompletion:
    params: dict[str, Any] = {"model": "gpt-4o-mini", "messages": MESSAGES, **kwargs}
    return cast(ChatCompletion, c.chat.completions.create(**params))


# -- off by default ---------------------------------------------------------------------


def test_no_key_means_no_cloud(server: FakeServer, cloud: FakeCloud) -> None:
    c = OpenAI(api_key="sk-test", http_client=server.http_client(), cache=True)
    ask(c)
    assert c.__dict__["_caelum_state"].cloud is None
    assert cloud.requests == []


def test_env_var_enables_cloud(
    server: FakeServer, cloud: FakeCloud, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAELUM_API_KEY", KEY)
    c = OpenAI(api_key="sk-test", http_client=server.http_client())
    assert c.__dict__["_caelum_state"].cloud is not None


def test_key_format_is_checked() -> None:
    with pytest.raises(ValueError, match="sk-caelum-"):
        resolve_cloud_key("sk-openai-123")


def test_repr_never_shows_the_key(cloud: FakeCloud) -> None:
    connection = CloudConnection(KEY + "-secret-part", base_url=cloud.url)
    assert "secret-part" not in repr(connection)
    connection.close()


# -- metrics ------------------------------------------------------------------------------


def test_usage_events_are_sent_without_prompts(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    ask(c)
    ask(c)  # local hit
    connection_of(c).flush()
    assert [e["cache"] for e in cloud.events] == ["miss", "exact"]
    miss = cloud.events[0]
    assert miss["input_tokens"] > 0 and miss["cost_usd"] > 0
    assert cloud.events[1]["saved_usd"] > 0
    sent = json.dumps(cloud.events)
    assert "Where is my order" not in sent and "answer #1" not in sent
    assert "sk-test" not in sent


def test_customer_label_is_attached(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    c = client(server, tmp_path)
    with caelum.customer("awa"):
        assert current_customer() == "awa"
        ask(c)
    ask(c)
    connection_of(c).flush()
    assert [e["customer"] for e in cloud.events] == ["awa", ""]
    with pytest.raises(ValueError):
        with caelum.customer(""):
            pass


def test_customer_is_isolated_between_async_tasks(server: FakeServer, cloud: FakeCloud) -> None:
    c = AsyncOpenAI(api_key="sk-test", http_client=server.async_http_client(), caelum_key=KEY)

    async def one(name: str) -> None:
        with caelum.customer(name):
            await asyncio.sleep(0)
            await c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)

    async def main() -> None:
        await asyncio.gather(one("awa"), one("koffi"))

    asyncio.run(main())
    connection_of(c).flush()
    assert sorted(e["customer"] for e in cloud.events) == ["awa", "koffi"]


def test_failed_requests_are_reported(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    server.failures = [400]
    c = client(server, tmp_path)
    with pytest.raises(openai.BadRequestError):
        ask(c)
    connection_of(c).flush()
    assert cloud.events[0]["cache"] == "error"


# -- shared cache ---------------------------------------------------------------------------


def test_shared_cache_serves_another_server(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    """Two app servers (two local caches) share answers through CAELUM Cloud."""
    first = client(server, tmp_path, cache=True, cache_path=tmp_path / "server1.sqlite")
    ask(first)
    connection_of(first).flush()
    assert len(cloud.cache) == 1
    second = client(server, tmp_path, cache=True, cache_path=tmp_path / "server2.sqlite")
    response = ask(second)
    assert caelum.cache_hit(response)
    assert response.choices[0].message.content == "answer #1"
    assert server.calls == 1  # the provider was called only once for both servers


def test_cloud_hit_is_copied_locally(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    first = client(server, tmp_path, cache=True, cache_path=tmp_path / "a.sqlite")
    ask(first)
    connection_of(first).flush()
    second = client(server, tmp_path, cache=True, cache_path=tmp_path / "b.sqlite")
    ask(second)
    gets = len([r for r in cloud.requests if r[0] == "GET"])
    ask(second)  # now local: no extra Cloud lookup
    assert len([r for r in cloud.requests if r[0] == "GET"]) == gets


def test_no_writable_home_still_uses_cloud_cache(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    """Serverless: the home folder is read-only, the shared cache still works."""
    unusable = tmp_path / "file-not-dir"
    unusable.write_text("x")
    first = client(server, tmp_path, cache=True, cache_path=unusable / "c.sqlite")
    ask(first)
    connection_of(first).flush()
    second = client(server, tmp_path, cache=True, cache_path=unusable / "c.sqlite")
    assert caelum.cache_hit(ask(second))


# -- the Cloud never breaks the app -------------------------------------------------------


@pytest.mark.parametrize("mode", ["down", "quota", "revoked"])
def test_cloud_problems_never_break_requests(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path, mode: str
) -> None:
    cloud.mode = mode
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    assert ask(c).choices[0].message.content == "answer #1"
    assert caelum.cache_hit(ask(c))  # the local cache keeps working
    connection_of(c).flush()


def test_slow_cloud_adds_at_most_the_lookup_timeout(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    cloud.mode = "slow"
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    connection_of(c).lookup_timeout = 0.2
    started = time.perf_counter()
    ask(c)
    assert time.perf_counter() - started < 1.5
    # After repeated failures the breaker opens: no more waiting at all.
    for i in range(3):
        ask(c, messages=[{"role": "user", "content": f"q{i}"}])
    started = time.perf_counter()
    ask(c, messages=[{"role": "user", "content": "fresh"}])
    assert time.perf_counter() - started < 0.15


def test_revoked_key_disables_cloud_with_one_warning(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cloud.mode = "revoked"
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    for i in range(3):
        ask(c, messages=[{"role": "user", "content": f"q{i}"}])
    connection_of(c).flush()
    assert caplog.text.count("invalid, revoked or expired CAELUM key") == 1
    seen = len(cloud.requests)
    ask(c, messages=[{"role": "user", "content": "more"}])
    connection_of(c).flush()
    assert len(cloud.requests) == seen  # no more calls with a dead key


def test_event_queue_is_bounded(cloud: FakeCloud) -> None:
    connection = CloudConnection(KEY, base_url=cloud.url)
    connection._closed = True  # pyright: ignore[reportPrivateUsage] - no background sending
    for _ in range(CloudConnection.MAX_QUEUE + 10):
        connection.record({"model": "m"})
    assert connection.dropped_events == 10


def test_one_connection_per_key(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    a = client(server, tmp_path)
    b = client(server, tmp_path)
    assert connection_of(a) is connection_of(b)


def test_slow_lookups_never_stop_uploads(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    # Far from the Cloud (e.g. West Africa) lookups can time out: the shared-cache writes and
    # the metrics must still be sent (they have their own breaker).
    cloud.mode = "slow_lookups"
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    connection_of(c).lookup_timeout = 0.2
    for i in range(5):
        ask(c, messages=[{"role": "user", "content": f"q{i}"}])
    connection_of(c).flush()
    assert len(cloud.events) == 5
    assert len(cloud.cache) == 5


def test_connections_are_reused(server: FakeServer, cloud: FakeCloud, tmp_path: Path) -> None:
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    for i in range(10):
        ask(c, messages=[{"role": "user", "content": f"q{i}"}])
    connection_of(c).flush()
    assert len(cloud.events) == 10
    # 10 lookups + writes + metrics over a handful of kept-alive connections, not ~25.
    assert cloud.connections <= 4


def test_server_closing_an_idle_connection_is_retried(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    c = client(server, tmp_path, cache=True, cache_path=tmp_path / "c.sqlite")
    ask(c)
    connection = connection_of(c)
    connection.flush()
    for conn in connection._pool:  # pyright: ignore[reportPrivateUsage]
        if conn.sock is not None:
            conn.sock.close()  # simulate the server dropping idle connections
    c2 = client(server, tmp_path, cache=True, cache_path=tmp_path / "other.sqlite")
    assert caelum.cache_hit(ask(c2))  # still served from the shared cache


# -- several servers of one customer, after the context keys of 0.2 -------------------------


def test_three_servers_of_one_customer_share_the_cloud_memory(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    servers = [
        client(server, tmp_path, cache=True, cache_path=tmp_path / f"server{i}.sqlite")
        for i in (1, 2, 3)
    ]
    with caelum.customer("boutique-kadi"):
        ask(servers[0])
        connection_of(servers[0]).flush()
        assert caelum.cache_hit(ask(servers[1]))
        assert caelum.cache_hit(ask(servers[2]))
    assert server.calls == 1  # one paid call for the three servers


def test_another_customer_sees_nothing_in_the_cloud_memory(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    first = client(server, tmp_path, cache=True, cache_path=tmp_path / "s1.sqlite")
    second = client(server, tmp_path, cache=True, cache_path=tmp_path / "s2.sqlite")
    with caelum.customer("boutique-kadi"):
        ask(first)
    connection_of(first).flush()
    with caelum.customer("faso-tech"):
        assert not caelum.cache_hit(ask(second))
    assert server.calls == 2


def test_frequency_admission_is_shared_between_servers(
    server: FakeServer, cloud: FakeCloud, tmp_path: Path
) -> None:
    # With CAELUM Cloud the repetitions are counted for the whole account (improvement B):
    # a question asked once on each of two servers is admitted.
    first = client(
        server, tmp_path, cache=True, cache_path=tmp_path / "n1.sqlite", cache_admission=2
    )
    second = client(
        server, tmp_path, cache=True, cache_path=tmp_path / "n2.sqlite", cache_admission=2
    )
    ask(first)  # 1st answer, counted in the Cloud: 1
    connection_of(first).flush()
    assert not caelum.cache_hit(ask(second))  # 2nd answer, on another server: count 2
    connection_of(second).flush()  # admitted: server 2 stores it, locally and in the Cloud
    connection_of(second).flush()
    third = client(
        server, tmp_path, cache=True, cache_path=tmp_path / "n3.sqlite", cache_admission=2
    )
    assert caelum.cache_hit(ask(third))  # a third server is served from the shared memory
    assert server.calls == 2


def test_cache_keys_carry_a_format_version() -> None:
    from caelum.cache import exact, semantic

    request = {"model": "gpt-4o", "messages": MESSAGES}
    key = exact.compute_cache_key(request, provider="openai", endpoint="chat", namespace="n")
    assert exact.KEY_VERSION == semantic.KEY_VERSION == 2
    original = exact.KEY_VERSION
    try:
        exact.KEY_VERSION = original + 1
        assert (
            exact.compute_cache_key(request, provider="openai", endpoint="chat", namespace="n")
            != key
        )
    finally:
        exact.KEY_VERSION = original
