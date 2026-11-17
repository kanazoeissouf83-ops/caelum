"""CAELUM Cloud connection (optional, off unless a CAELUM key is given).

With ``caelum_key="sk-caelum-..."`` (or ``$CAELUM_API_KEY``) a client:

* sends **usage metrics** to CAELUM Cloud in the background (model, tokens, cost, cache
  hit or miss, latency, and the end customer set with :func:`customer`). Prompts and
  answers are never part of the metrics;
* with ``cache=True``, also uses a **shared cache** in CAELUM Cloud behind the local one,
  so that every server of an app (or every short-lived serverless instance) reuses the
  same answers. Cached answers are stored in the customer's own, isolated account.

Your AI calls never go through CAELUM Cloud: the provider is always called directly with
your own key. The Cloud is never on the critical path: every Cloud call has a short
timeout, failures switch the Cloud off for a while (circuit breaker), and the app simply
continues with its local cache. Only the standard library is used (no new dependency).
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import hashlib
import http.client
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, cast

from caelum.cache.stores import CacheStore
from caelum.utils import logger

__all__ = [
    "DEFAULT_CLOUD_URL",
    "CloudConnection",
    "TieredStore",
    "current_customer",
    "customer",
    "resolve_cloud_key",
]

DEFAULT_CLOUD_URL = "https://api.caelum-ai.com"
KEY_PREFIX = "sk-caelum-"

_customer: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "caelum_customer", default=None
)


@contextlib.contextmanager
def customer(name: str) -> Generator[None]:
    """Attribute the requests made inside the block to one end customer.

    Shown as "cost per customer" in the CAELUM Cloud dashboard. Works with threads and
    asyncio (each task keeps its own value)::

        with caelum.customer("awa"):
            client.chat.completions.create(...)

    IMPORTANT — isolation between your end users. The cache key includes this id, so two
    different customers never share a cached answer. If your app serves several end users
    and you do **not** wrap each one in ``caelum.customer(...)``, they all fall into the
    same bucket, and a cacheable answer for one can be served to another. Wrapping each end
    user is therefore required whenever an answer could be specific to that user. (For a
    public FAQ where every answer is the same for everyone, a shared bucket is fine.)

    Use an internal id rather than personal data (an email, a name): it is sent to CAELUM
    Cloud as a metric label and shown in your dashboard.
    """
    if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("customer name must be a non-empty string")
    token = _customer.set(name[:200])
    try:
        yield
    finally:
        _customer.reset(token)


def current_customer() -> str | None:
    """End customer set by the innermost :func:`customer` block, if any."""
    return _customer.get()


def resolve_cloud_key(explicit: str | None) -> str | None:
    """The CAELUM key to use: the explicit argument, else ``$CAELUM_API_KEY``."""
    key = explicit if explicit is not None else os.environ.get("CAELUM_API_KEY") or None
    if key is None:
        return None
    if not isinstance(key, str) or not key.startswith(KEY_PREFIX):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("caelum_key must be a CAELUM key starting with 'sk-caelum-'")
    return key


@dataclass
class _Breaker:
    """Switches the Cloud off for a while after failures, so the app never waits on it."""

    cooldown: float = 60.0
    failures_to_open: int = 3
    failures: int = 0
    open_until: float = 0.0
    disabled: bool = False  # permanent (invalid key)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def allows(self) -> bool:
        return not self.disabled and time.monotonic() >= self.open_until

    def success(self) -> None:
        with self.lock:
            self.failures = 0

    def failure(self, *, now: bool = False) -> None:
        with self.lock:
            self.failures += 1
            if now or self.failures >= self.failures_to_open:
                self.open_until = time.monotonic() + self.cooldown
                self.failures = 0


class CloudConnection:
    """HTTP connection to CAELUM Cloud, shared by every client copy.

    Args:
        api_key: The ``sk-caelum-...`` key of the account.
        base_url: API address (default :data:`DEFAULT_CLOUD_URL` or ``$CAELUM_CLOUD_URL``).
        lookup_timeout: Max seconds a cache lookup may add to a request (default 1 s;
            with a reused connection a lookup usually takes 0.1-0.3 s, even from Africa).
        flush_interval: Seconds between background metric uploads.
    """

    MAX_QUEUE = 10_000
    BATCH = 200

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        lookup_timeout: float = 1.0,
        flush_interval: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = (
            base_url or os.environ.get("CAELUM_CLOUD_URL") or DEFAULT_CLOUD_URL
        ).rstrip("/")
        self.lookup_timeout = lookup_timeout
        self.flush_interval = flush_interval
        self.breaker = _Breaker()  # background writes and metrics
        # WHY: a slow lookup must only pause lookups, never the background uploads.
        self.lookup_breaker = _Breaker()
        self._pool: list[http.client.HTTPConnection] = []  # idle keep-alive connections
        self._pool_lock = threading.Lock()
        self._flush_lock = threading.Lock()  # one upload at a time (thread vs. close())
        self._events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.MAX_QUEUE)
        self._writes: queue.Queue[tuple[str, str, int | None]] = queue.Queue(maxsize=1_000)
        # Shared frequency admission: (fingerprint, tag, n, ttl, callback(admitted or None)).
        self._admissions: queue.Queue[tuple[str, str, int, int, Callable[[bool | None], None]]] = (
            queue.Queue(maxsize=1_000)
        )
        # The shared counter is a Starter feature: after a "plan_required" answer, stop
        # asking for an hour (the local counter is used meanwhile).
        self._admission_off_until = 0.0
        self._wakeup = threading.Event()
        self._closed = False
        self._pid = os.getpid()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped_events = 0
        _connections.add(self)

    def __repr__(self) -> str:
        # Never show any of the secret: only the non-secret prefix ("sk-caelum-").
        return f"<caelum.CloudConnection {self.base_url} key={self.api_key[: len(KEY_PREFIX)]}...>"

    # -- HTTP ----------------------------------------------------------------------

    def _new_connection(self, timeout: float) -> http.client.HTTPConnection:
        url = urllib.parse.urlsplit(self.base_url)
        host = url.hostname or ""
        if url.scheme == "https":
            return http.client.HTTPSConnection(host, url.port, timeout=timeout)
        return http.client.HTTPConnection(host, url.port, timeout=timeout)

    def _request(
        self, method: str, path: str, body: bytes | None, timeout: float
    ) -> tuple[int, bytes]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "caelum-python",
        }
        if urllib.request.getproxies().get(urllib.parse.urlsplit(self.base_url).scheme):
            return self._request_via_urllib(method, path, body, timeout, headers)
        prefix = urllib.parse.urlsplit(self.base_url).path.rstrip("/")
        # WHY: a new TLS connection costs 0.3-0.6 s far from Cloudflare (e.g. West Africa);
        # reusing open connections keeps a lookup around 0.1-0.3 s.
        for attempt in range(2):
            with self._pool_lock:
                conn = self._pool.pop() if self._pool else None
            reused = conn is not None
            if conn is None:
                conn = self._new_connection(timeout)
            try:
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                conn.request(method, prefix + path, body=body, headers=headers)
                response = conn.getresponse()
                data = response.read()
            except (http.client.HTTPException, OSError) as exc:
                conn.close()
                if reused and attempt == 0 and not isinstance(exc, TimeoutError):
                    continue  # an idle connection was dropped: retry once on a fresh one
                raise
            except BaseException:
                conn.close()
                raise
            if response.will_close:
                conn.close()
            else:
                with self._pool_lock:
                    if len(self._pool) < 4 and not self._closed:
                        self._pool.append(conn)
                    else:
                        conn.close()
            return response.status, data
        raise ConnectionError("unreachable")  # pragma: no cover

    def _request_via_urllib(
        self, method: str, path: str, body: bytes | None, timeout: float, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        """Behind an HTTP(S) proxy: urllib honours the proxy settings (no reuse)."""
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read() if exc.fp else b""

    def warm_up(self) -> None:
        """Open one connection ahead of the first lookup (called in the background)."""
        if self.lookup_breaker.allows():
            with contextlib.suppress(Exception):
                self._request("GET", "/v1/health", None, 5.0)

    def _call(
        self,
        method: str,
        path: str,
        body: bytes | None,
        timeout: float,
        breaker: _Breaker | None = None,
    ) -> tuple[int, bytes] | None:
        """One Cloud call through a circuit breaker (None = skipped or failed)."""
        breaker = breaker or self.breaker
        if not breaker.allows():
            return None
        try:
            status, data = self._request(method, path, body, timeout)
        except Exception as exc:  # network error, timeout, DNS...
            logger.debug("caelum: cloud %s %s failed: %s", method, path, exc)
            breaker.failure()
            return None
        if status == 401:
            if not self.breaker.disabled:
                logger.warning(
                    "caelum: invalid, revoked or expired CAELUM key, CAELUM Cloud disabled"
                )
            self.breaker.disabled = True
            self.lookup_breaker.disabled = True
            return None
        if status == 429:
            # Monthly limit reached: the shared cache is paused, the app goes on locally.
            breaker.failure(now=True)
            return status, data
        if status >= 500:
            breaker.failure()
            return None
        breaker.success()
        return status, data

    # -- shared cache --------------------------------------------------------------

    def cache_get(self, key: str, tag: str = "-") -> Any | None:
        result = self._call(
            "GET", f"/v1/cache/{key}?c={tag}", None, self.lookup_timeout, self.lookup_breaker
        )
        if result is None or result[0] != 200:
            return None
        try:
            return json.loads(result[1])
        except ValueError:
            return None

    def cache_put(self, key: str, value: Any, ttl: float | None, tag: str = "-") -> None:
        """Queue a write (sent in the background: never slows the request down)."""
        try:
            encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return
        try:
            self._writes.put_nowait((f"{key}?c={tag}", encoded, None if ttl is None else int(ttl)))
        except queue.Full:
            return
        self._start()
        # WHY: other servers can only reuse the answer once it is uploaded: send it now.
        self._wakeup.set()

    def admission_observe(
        self,
        fingerprint: str,
        tag: str,
        n: int,
        ttl: float,
        callback: Callable[[bool | None], None],
    ) -> bool:
        """Count one AI answer of a question for the whole account, in the background.

        ``callback(admitted)`` runs later with True (cache it now), False (not yet) or
        None (Cloud unreachable). Returns False when the queue is full (nothing sent).
        """
        if time.monotonic() < self._admission_off_until:
            return False  # not included in this account's plan: local counter only
        try:
            self._admissions.put_nowait((fingerprint, tag, n, int(ttl), callback))
        except queue.Full:
            return False
        self._start()
        self._wakeup.set()
        return True

    def admission_pending(self, tag: str) -> int | None:
        """Shared candidates still waiting for repetitions (None if unreachable)."""
        result = self._call("GET", f"/v1/admission?c={tag}", None, 5.0)
        if result is None or result[0] != 200:
            return None
        try:
            return int(json.loads(result[1]).get("pending", 0))
        except (ValueError, AttributeError, TypeError):
            return None

    def cache_clear(self, tag: str, older_than: float | None = None) -> int | None:
        """Delete one end customer's entries from the shared memory (synchronous).

        Returns the number deleted, or None if the Cloud could not be reached.
        """
        self.flush(timeout=5.0)  # pending writes first, so they are cleared too
        deleted = 0
        query = f"?c={tag}" + (f"&older_than={int(older_than)}" if older_than is not None else "")
        for _ in range(50):  # at most 50 x 500 entries per call
            result = self._call("DELETE", f"/v1/cache{query}", None, 10.0)
            if result is None or result[0] != 200:
                return None
            try:
                body = json.loads(result[1])
            except ValueError:
                return None
            deleted += int(body.get("deleted", 0))
            if not body.get("remaining"):
                break
        return deleted

    # -- metrics -------------------------------------------------------------------

    def record(self, event: dict[str, Any]) -> None:
        """Queue one usage event (dropped, never blocking, if the queue is full)."""
        try:
            self._events.put_nowait(event)
        except queue.Full:
            self.dropped_events += 1
            return
        self._start()
        if self._events.qsize() >= self.BATCH:
            self._wakeup.set()

    # -- background thread ---------------------------------------------------------

    def _start(self) -> None:
        if os.getpid() != self._pid:
            # WHY: after fork the worker thread does not exist in the child.
            self._pid = os.getpid()
            self._thread = None
        if self._thread is not None or self._closed:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="caelum-cloud", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        self.warm_up()
        while not self._closed:
            self._wakeup.wait(self.flush_interval)
            self._wakeup.clear()
            self.flush(timeout=5.0)

    def flush(self, timeout: float = 5.0) -> None:
        """Send queued cache writes and metrics now."""
        with self._flush_lock:
            self._flush(timeout)

    def _flush(self, timeout: float) -> None:
        while True:
            try:
                key, encoded, ttl = self._writes.get_nowait()
            except queue.Empty:
                break
            query = f"&ttl={ttl}" if ttl else ""
            self._call("PUT", f"/v1/cache/{key}{query}", encoded.encode("utf-8"), timeout)
        while True:
            try:
                fingerprint, tag, n, ttl, callback = self._admissions.get_nowait()
            except queue.Empty:
                break
            body = json.dumps({"fingerprint": fingerprint, "n": n, "ttl": ttl}).encode("utf-8")
            result = self._call("POST", f"/v1/admission?c={tag}", body, timeout)
            admitted: bool | None = None
            if result is not None and result[0] == 403:
                self._admission_off_until = time.monotonic() + 3600
            if result is not None and result[0] == 200:
                try:
                    admitted = bool(json.loads(result[1]).get("admitted"))
                except (ValueError, AttributeError):
                    admitted = None
            try:
                callback(admitted)
            except Exception:  # pragma: no cover - a callback must never stop uploads
                logger.warning("caelum: shared admission callback failed", exc_info=True)
        while True:
            batch: list[dict[str, Any]] = []
            while len(batch) < self.BATCH:
                try:
                    batch.append(self._events.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                return
            body = json.dumps({"events": batch}, separators=(",", ":")).encode("utf-8")
            if self._call("POST", "/v1/events", body, timeout) is None:
                self.dropped_events += len(batch)  # Cloud unreachable: metrics are best effort
                return

    def close(self) -> None:
        """Send what is queued (briefly) and stop the background thread."""
        if self._closed:
            return
        self._closed = True
        self._wakeup.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            # WHY: let an upload in progress finish; the process may exit right after.
            thread.join(timeout=3.0)
        try:
            self.flush(timeout=2.0)
        except Exception:  # pragma: no cover - best effort on shutdown
            pass
        with self._pool_lock:
            pool, self._pool = self._pool, []
        for conn in pool:
            conn.close()


_connections: weakref.WeakSet[CloudConnection] = weakref.WeakSet()


@atexit.register
def _flush_at_exit() -> None:  # pragma: no cover - runs at interpreter shutdown
    for connection in list(_connections):
        connection.close()


def customer_tag(customer: str) -> str:
    """Short hash of an end customer in Cloud keys ("-" for none): the name never leaves."""
    if not customer:
        return "-"
    return hashlib.sha256(customer.encode("utf-8")).hexdigest()[:16]


class TieredStore(CacheStore):
    """Local cache first, then the shared CAELUM Cloud cache.

    A local hit costs nothing extra. A local miss asks the Cloud (bounded by the lookup
    timeout); a Cloud hit is copied locally. Writes go to both, the Cloud one in the
    background.
    """

    def __init__(self, local: CacheStore, cloud: CloudConnection, ttl: float | None) -> None:
        self.local = local
        self.cloud = cloud
        self.ttl = ttl

    supports_owner = True

    def get(self, key: str, max_age: float | None = None, owner: str = "") -> Any | None:
        if getattr(self.local, "supports_owner", False):
            value = cast(Any, self.local).get(key, max_age=max_age)
        else:
            value = self.local.get(key)
        if value is not None:
            return value
        remote = self.cloud.cache_get(key, customer_tag(owner))
        if remote is None:
            return None
        if not isinstance(remote, dict):
            return None
        envelope = cast("dict[str, Any]", remote)
        payload: Any = envelope.get("payload")
        if payload is None:
            return None
        ttl = envelope.get("ttl", self.ttl)
        lifetime = ttl if isinstance(ttl, (int, float)) else self.ttl
        created = envelope.get("created")
        if isinstance(created, (int, float)):
            # Same rule as locally: expiry = created + lifetime, and a shortened lifetime
            # (max_age) applies to entries already stored.
            age = time.time() - created
            if max_age is not None and age >= max_age:
                return None
            if lifetime is not None:
                lifetime = lifetime - age
                if lifetime <= 0:
                    return None
        self._set_local(key, payload, lifetime, owner)
        return payload

    def _set_local(self, key: str, value: Any, ttl: float | None, owner: str) -> None:
        if getattr(self.local, "supports_owner", False):
            cast(Any, self.local).set(key, value, ttl, owner=owner)
        else:
            self.local.set(key, value, ttl)

    def set(self, key: str, value: Any, ttl: float | None, owner: str = "") -> None:
        self._set_local(key, value, ttl, owner)
        envelope = {"payload": value, "ttl": ttl, "created": time.time()}
        self.cloud.cache_put(key, envelope, ttl, customer_tag(owner))

    def clear_customer(self, customer: str, older_than: float | None = None) -> int:
        """Delete one end customer's entries locally and in the shared memory."""
        local = 0
        if getattr(self.local, "supports_owner", False):
            local = cast(Any, self.local).clear_customer(customer, older_than)
        remote = self.cloud.cache_clear(customer_tag(customer), older_than)
        return local + (remote or 0)

    def delete(self, key: str) -> None:
        self.local.delete(key)

    def clear(self) -> None:
        self.local.clear()

    def close(self) -> None:
        self.local.close()
