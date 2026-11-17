"""Cache storage backends.

The default backend is SQLite through Python's standard ``sqlite3`` module: zero
configuration, no server, no extra dependency. Values are stored as JSON text — never
``pickle``, which could execute arbitrary code when a cache file is tampered with.

Concurrency design (benchmark: a single locked connection added +53 ms at p95 with
8 concurrent threads on Windows):

* **writes never block a request**: ``set`` records the entry in an in-memory pending
  map and a background thread commits pending entries in batches;
* **reads never wait for writes**: each thread reads through its own connection (SQLite
  WAL mode allows readers during a write), after checking the pending map, so a value is
  visible to the same process immediately after ``set``.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import sqlite3
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caelum.utils import logger

__all__ = ["CacheStore", "SQLiteStore", "caelum_home", "default_cache_path"]

SCHEMA_VERSION = "1"
MEMORY = ":memory:"
_memory_ids = itertools.count()


def caelum_home() -> Path:
    """Directory where CAELUM keeps local data (``$CAELUM_HOME`` or ``~/.caelum``)."""
    configured = os.environ.get("CAELUM_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".caelum"


def default_cache_path() -> Path:
    """Default location of the local cache database."""
    return caelum_home() / "cache.sqlite"


def prepare_private_file(location: str) -> None:
    """Create the cache folder and file readable by the current user only.

    The cache holds prompts and answers, which may contain personal data. On Linux and
    macOS the default permissions would let every local user read it, so the folder is
    created with mode 0700 and the file with mode 0600 (SQLite gives its ``-wal`` and
    ``-shm`` files the same permissions as the database). Existing files are left as
    they are. On Windows, files in the user profile are already private.
    """
    path = Path(location)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix" and not path.exists():
        try:
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        except OSError:  # pragma: no cover - let sqlite report the real problem
            pass


class CacheStore(ABC):
    """Key-value store holding JSON-compatible values with an optional TTL."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the value for ``key``, or None if missing or expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None) -> None:
        """Store ``value`` under ``key`` for ``ttl`` seconds (None = never expires)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the store."""

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None


@dataclass(frozen=True)
class _Pending:
    """A write not yet committed. ``encoded`` None means "delete"."""

    encoded: str | None
    created_at: float
    expires_at: float | None
    owner: str = ""  # end customer of the entry ("" = none)


PURGE_INTERVAL = 600.0  # seconds between automatic removals of expired entries


class _Backend:
    """Connections, pending writes and the writer thread of one store.

    Kept separate from :class:`SQLiteStore` so that a ``weakref.finalize`` callback can
    flush and close everything when the store is garbage-collected.
    """

    def __init__(
        self,
        uri: str,
        *,
        use_uri: bool,
        persistent: bool,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.uri = uri
        self.clock = clock
        self.use_uri = use_uri
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending: dict[str, _Pending] = {}
        self.readers: list[sqlite3.Connection] = []
        self.local = threading.local()
        self.wakeup = threading.Event()
        self.closed = False
        self.pid = os.getpid()
        self.thread: threading.Thread | None = None
        self.last_purge = clock()  # then every PURGE_INTERVAL, on a background write
        self.writer = self.connect()
        try:
            self._init_schema(persistent)
        except BaseException:
            # WHY: a corrupted or unreadable file fails here: never leak the connection.
            self.writer.close()
            raise

    def _init_schema(self, persistent: bool) -> None:
        with self.write_lock:
            if persistent:
                self.writer.execute("PRAGMA journal_mode=WAL")
            self.writer.execute("PRAGMA synchronous=NORMAL")
            self.writer.execute(
                "CREATE TABLE IF NOT EXISTS caelum_cache ("
                " key TEXT PRIMARY KEY,"
                " value TEXT NOT NULL,"
                " created_at REAL NOT NULL,"
                " expires_at REAL)"
            )
            self.writer.execute(
                "CREATE INDEX IF NOT EXISTS caelum_cache_expires ON caelum_cache (expires_at)"
            )
            # WHY: per end customer clearing (and lifetimes) need to know whose entry it is.
            columns = {row[1] for row in self.writer.execute("PRAGMA table_info(caelum_cache)")}
            if "customer" not in columns:
                self.writer.execute(
                    "ALTER TABLE caelum_cache ADD COLUMN customer TEXT NOT NULL DEFAULT ''"
                )
            self.writer.execute(
                "CREATE TABLE IF NOT EXISTS caelum_meta (name TEXT PRIMARY KEY, value TEXT)"
            )
            self.writer.execute(
                "INSERT OR IGNORE INTO caelum_meta (name, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.uri, uri=self.use_uri, check_same_thread=False, isolation_level=None
        )
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def reader(self) -> sqlite3.Connection:
        """This thread's read connection (created on first use)."""
        conn: sqlite3.Connection | None = getattr(self.local, "conn", None)
        if conn is None:
            conn = self.connect()
            self.local.conn = conn
            with self.pending_lock:
                self.readers.append(conn)
        return conn

    def start_writer(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, name="caelum-cache-writer", daemon=True)
            self.thread.start()

    def run(self) -> None:
        while True:
            self.wakeup.wait()
            # WHY: clear BEFORE checking `closed`. Checking first could miss the wake-up
            # sent by close() (it would be cleared right after), leaving close() to wait.
            self.wakeup.clear()
            if self.closed:
                return
            time.sleep(0.002)  # coalesce bursts of writes into one transaction
            try:
                self.flush()
            except Exception:  # pragma: no cover - never kill the writer thread
                logger.warning("caelum: background cache write failed", exc_info=True)

    def flush(self) -> None:
        """Commit every pending write (called by the writer thread, or explicitly)."""
        with self.write_lock:
            with self.pending_lock:
                batch = dict(self.pending)
            if not batch:
                return
            self.writer.execute("BEGIN")
            try:
                for key, entry in batch.items():
                    if entry.encoded is None:
                        self.writer.execute("DELETE FROM caelum_cache WHERE key = ?", (key,))
                    else:
                        self.writer.execute(
                            "INSERT OR REPLACE INTO caelum_cache"
                            " (key, value, created_at, expires_at, customer)"
                            " VALUES (?, ?, ?, ?, ?)",
                            (key, entry.encoded, entry.created_at, entry.expires_at, entry.owner),
                        )
                self.writer.execute("COMMIT")
            except BaseException:
                self.writer.execute("ROLLBACK")
                raise
            with self.pending_lock:
                for key, entry in batch.items():
                    if self.pending.get(key) is entry:  # not overwritten meanwhile
                        del self.pending[key]
            # WHY: expired answers are never served but would stay on disk forever; a busy
            # server would see its cache file grow without limit.
            now = self.clock()
            if now - self.last_purge >= PURGE_INTERVAL:
                self.last_purge = now
                self.writer.execute(
                    "DELETE FROM caelum_cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.flush()
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.warning("caelum: could not flush the cache on close", exc_info=True)
        self.closed = True
        self.wakeup.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        with self.pending_lock:
            readers, self.readers = self.readers, []
        for conn in [*readers, self.writer]:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass


_live_backends: weakref.WeakSet[_Backend] = weakref.WeakSet()


@atexit.register
def _flush_at_exit() -> None:  # pragma: no cover - runs at interpreter shutdown
    for backend in list(_live_backends):
        backend.close()


class SQLiteStore(CacheStore):
    """Thread-safe SQLite key-value store with per-entry expiration.

    Args:
        path: Database file. Defaults to :func:`default_cache_path`. Use ``":memory:"``
            for a process-local, non-persistent store.
        clock: Returns the current time in seconds (injectable for tests).
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        location = str(path) if path is not None else str(default_cache_path())
        self.path = location
        self._clock = clock
        self._open()

    def _open(self) -> None:
        location = self.path
        if location == MEMORY:
            # WHY: a named shared in-memory database, so that every per-thread connection
            # sees the same data (a plain ":memory:" is private to one connection).
            uri = f"file:caelum-memory-{os.getpid()}-{next(_memory_ids)}?mode=memory&cache=shared"
            self._live = _Backend(uri, use_uri=True, persistent=False, clock=self._clock)
        else:
            prepare_private_file(location)
            self._live = _Backend(location, use_uri=False, persistent=True, clock=self._clock)
        _live_backends.add(self._live)
        self._finalizer = weakref.finalize(self, self._live.close)

    @property
    def _backend(self) -> _Backend:
        """The store backend, reopened after ``os.fork()``.

        WHY: a forked child (gunicorn/uwsgi workers, multiprocessing) inherits the
        parent's SQLite connections, which SQLite forbids using, and a writer thread that
        does not exist in the child. The child abandons them and opens its own.
        """
        backend = self._live
        if backend.pid != os.getpid():
            self._finalizer.detach()
            _live_backends.discard(backend)
            self._open()
            backend = self._live
        return backend

    @property
    def _conn(self) -> sqlite3.Connection:
        """Writer connection (for maintenance and tests)."""
        return self._backend.writer

    supports_owner = True  # get(max_age=...), set(owner=...), clear_customer()

    def get(self, key: str, max_age: float | None = None, owner: str = "") -> Any | None:
        """The value of ``key``, or None if missing, expired, or older than ``max_age``.

        ``max_age`` applies a lifetime shortened after the entry was written.
        """
        backend = self._backend
        now = self._clock()
        with backend.pending_lock:
            entry = backend.pending.get(key)
        if entry is not None:
            if entry.encoded is None or (entry.expires_at is not None and entry.expires_at <= now):
                return None
            if max_age is not None and entry.created_at + max_age <= now:
                return None
            return json.loads(entry.encoded)
        row = (
            backend.reader()
            .execute("SELECT value, expires_at, created_at FROM caelum_cache WHERE key = ?", (key,))
            .fetchone()
        )
        if row is None:
            return None
        value, expires_at, created_at = row
        if expires_at is not None and expires_at <= now:
            return None  # removed later by purge_expired()
        if max_age is not None and created_at + max_age <= now:
            return None  # the customer's lifetime was shortened since
        return json.loads(value)

    def _enqueue(self, key: str, entry: _Pending) -> None:
        backend = self._backend
        if backend.closed:
            return  # writes after close() are dropped instead of piling up in memory
        with backend.pending_lock:
            backend.pending[key] = entry
        backend.start_writer()
        backend.wakeup.set()

    def set(self, key: str, value: Any, ttl: float | None, owner: str = "") -> None:
        now = self._clock()
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        self._enqueue(key, _Pending(encoded, now, None if ttl is None else now + ttl, owner))

    def clear_customer(self, customer: str, older_than: float | None = None) -> int:
        """Delete the entries of one end customer (only those older than ``older_than``
        seconds when given). Returns the number of entries deleted."""
        self.flush()
        backend = self._backend
        with backend.write_lock:
            if older_than is None:
                cursor = backend.writer.execute(
                    "DELETE FROM caelum_cache WHERE customer = ?", (customer,)
                )
            else:
                cursor = backend.writer.execute(
                    "DELETE FROM caelum_cache WHERE customer = ? AND created_at <= ?",
                    (customer, self._clock() - older_than),
                )
            return cursor.rowcount

    def delete(self, key: str) -> None:
        self._enqueue(key, _Pending(None, self._clock(), None))

    def flush(self) -> None:
        """Commit pending writes now (they are committed in the background otherwise)."""
        self._backend.flush()

    def clear(self) -> None:
        backend = self._backend
        with backend.pending_lock:
            backend.pending.clear()
        with backend.write_lock:
            backend.writer.execute("DELETE FROM caelum_cache")

    def purge_expired(self) -> int:
        """Delete expired entries and return how many were removed."""
        self.flush()
        with self._backend.write_lock:
            cursor = self._backend.writer.execute(
                "DELETE FROM caelum_cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (self._clock(),),
            )
            return cursor.rowcount

    def __len__(self) -> int:
        self.flush()
        with self._backend.write_lock:
            (count,) = self._backend.writer.execute("SELECT COUNT(*) FROM caelum_cache").fetchone()
        return int(count)

    def close(self) -> None:
        """Commit pending writes and close every connection."""
        self._finalizer()
