"""Local vector store for the semantic cache (SQLite + in-memory index).

Embeddings are stored L2-normalized as float32 blobs (standard ``array`` module, no
pickle). Similarity is the dot product of normalized vectors, i.e. the cosine.

Search is a brute-force scan restricted to entries sharing the same *context key* (same
model, system prompt, history and parameters) and the same embedding model, so the
candidate set stays small. Vectors are kept in memory per context after first use.

Speed: uses ``numpy`` when installed (``pip install "caelum-ai[fast]"``), otherwise
``math.sumprod`` (Python 3.12+) or a pure-Python dot product.

Scale-up path (corpus > 100k embeddings): approximate nearest-neighbour index (FAISS,
sqlite-vec) locally, Cloudflare Vectorize in CAELUM Cloud.
"""

from __future__ import annotations

import json
import math
import operator
import os
import sqlite3
import threading
import time
import weakref
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from caelum.cache.stores import MEMORY, default_cache_path, prepare_private_file

__all__ = ["SQLiteVectorStore", "VectorMatch", "normalize"]

_np: Any
try:  # optional acceleration
    import numpy as _np  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - depends on the environment
    _np = None

_sumprod: Callable[[Sequence[float], Sequence[float]], float] | None = getattr(
    math, "sumprod", None
)

DEFAULT_MAX_ENTRIES = 10_000


def normalize(vector: Sequence[float]) -> array[float]:
    """Return ``vector`` scaled to unit length, as a float32 array.

    Raises:
        ValueError: If the vector is empty or has zero norm.
    """
    if not vector:
        raise ValueError("empty embedding vector")
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("embedding vector has zero or non-finite norm")
    return array("f", (x / norm for x in vector))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if _sumprod is not None:
        return _sumprod(a, b)
    return math.fsum(map(operator.mul, a, b))


@dataclass(frozen=True)
class VectorMatch:
    """Best match returned by :meth:`SQLiteVectorStore.search`."""

    value: Any
    score: float
    query: str | None = None  # stored question text (None for entries written before 0.1)
    row_id: int | None = None  # identifies the entry (to replace or delete it)


@dataclass
class _Entry:
    row_id: int
    vector: array[float]
    expires_at: float | None
    created_at: float = 0.0


class SQLiteVectorStore:
    """Semantic cache storage.

    Args:
        path: Database file (default: the CAELUM cache file). ``":memory:"`` for tests.
        max_entries: Maximum number of stored embeddings; oldest entries are evicted.
        clock: Returns the current time in seconds (injectable for tests).
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        location = str(path) if path is not None else str(default_cache_path())
        if location != MEMORY:
            prepare_private_file(location)
        self.path = location
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._connection = sqlite3.connect(location, check_same_thread=False, isolation_level=None)
        self._finalizer = weakref.finalize(self, self._connection.close)
        # (context_key, embedder) -> entries, loaded lazily.
        self._index: dict[tuple[str, str], list[_Entry]] = {}
        with self._lock:
            self._init_schema()

    def _init_schema(self) -> None:
        """Create or migrate the table (caller holds the lock)."""
        if self.path != MEMORY:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS caelum_semantic ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " context_key TEXT NOT NULL,"
            " embedder TEXT NOT NULL,"
            " vector BLOB NOT NULL,"
            " value TEXT NOT NULL,"
            " created_at REAL NOT NULL,"
            " expires_at REAL)"
        )
        # WHY: the question text is needed by the LLM verifier. Older cache files
        # lack the column; their entries simply cannot be verified (never reused).
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(caelum_semantic)")}
        if "query" not in columns:
            self._conn.execute("ALTER TABLE caelum_semantic ADD COLUMN query TEXT")
        if "customer" not in columns:
            self._conn.execute(
                "ALTER TABLE caelum_semantic ADD COLUMN customer TEXT NOT NULL DEFAULT ''"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS caelum_semantic_context"
            " ON caelum_semantic (context_key, embedder)"
        )
        (self._count,) = self._conn.execute("SELECT COUNT(*) FROM caelum_semantic").fetchone()

    # -- internals -------------------------------------------------------------------

    def _load(self, context_key: str, embedder: str) -> list[_Entry]:
        # Caller holds the lock.
        cached = self._index.get((context_key, embedder))
        if cached is None:
            rows = self._conn.execute(
                "SELECT id, vector, expires_at, created_at FROM caelum_semantic"
                " WHERE context_key = ? AND embedder = ?",
                (context_key, embedder),
            ).fetchall()
            bucket: list[_Entry] = []
            for row_id, blob, expires_at, created_at in rows:
                vector = array("f")
                vector.frombytes(blob)
                bucket.append(_Entry(int(row_id), vector, expires_at, float(created_at)))
            self._index[(context_key, embedder)] = bucket
            return bucket
        return cached

    def _evict_if_needed(self) -> None:
        # Caller holds the lock.
        if self._count <= self.max_entries:
            return
        excess = self._count - self.max_entries
        self._conn.execute(
            "DELETE FROM caelum_semantic WHERE id IN"
            " (SELECT id FROM caelum_semantic ORDER BY id LIMIT ?)",
            (excess,),
        )
        (self._count,) = self._conn.execute("SELECT COUNT(*) FROM caelum_semantic").fetchone()
        self._index.clear()  # reloaded lazily from the database

    # -- public API ------------------------------------------------------------------

    def add(
        self,
        context_key: str,
        embedder: str,
        vector: Sequence[float],
        value: Any,
        ttl: float | None,
        query: str | None = None,
        replace_id: int | None = None,
        customer: str = "",
    ) -> None:
        """Store a normalized embedding, its JSON-compatible value and question text.

        ``replace_id`` removes that older entry in the same step (gray-zone refresh).
        """
        normalized = normalize(vector)
        now = self._clock()
        expires_at = None if ttl is None else now + ttl
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            if replace_id is not None:
                self._delete_locked(replace_id)
            cursor = self._conn.execute(
                "INSERT INTO caelum_semantic"
                " (context_key, embedder, vector, value, created_at, expires_at, query, customer)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    context_key,
                    embedder,
                    normalized.tobytes(),
                    encoded,
                    now,
                    expires_at,
                    query,
                    customer,
                ),
            )
            self._count += 1
            bucket = self._index.get((context_key, embedder))
            if bucket is not None and cursor.lastrowid is not None:
                bucket.append(_Entry(cursor.lastrowid, normalized, expires_at, now))
            self._evict_if_needed()

    def _delete_locked(self, row_id: int) -> None:
        # Caller holds the lock.
        deleted = self._conn.execute("DELETE FROM caelum_semantic WHERE id = ?", (row_id,))
        if deleted.rowcount:
            self._count -= deleted.rowcount
            for bucket in self._index.values():
                bucket[:] = [entry for entry in bucket if entry.row_id != row_id]

    def delete(self, row_id: int) -> None:
        """Remove one entry (no error if it is already gone)."""
        with self._lock:
            self._delete_locked(row_id)

    def delete_customer(self, customer: str, older_than: float | None = None) -> int:
        """Delete the entries of one end customer (only those older than ``older_than``
        seconds when given). Returns the number of entries deleted."""
        with self._lock:
            if older_than is None:
                cursor = self._conn.execute(
                    "DELETE FROM caelum_semantic WHERE customer = ?", (customer,)
                )
            else:
                cursor = self._conn.execute(
                    "DELETE FROM caelum_semantic WHERE customer = ? AND created_at <= ?",
                    (customer, self._clock() - older_than),
                )
            if cursor.rowcount:
                self._count -= cursor.rowcount
                self._index.clear()  # reloaded lazily from the database
            return cursor.rowcount

    def search(
        self,
        context_key: str,
        embedder: str,
        vector: Sequence[float],
        threshold: float,
        max_age: float | None = None,
    ) -> VectorMatch | None:
        """Return the most similar live entry with cosine >= ``threshold``, if any.

        ``max_age`` ignores entries older than that many seconds (a shortened lifetime).
        """
        query = normalize(vector)
        now = self._clock()
        oldest = None if max_age is None else now - max_age
        with self._lock:
            entries = [
                entry
                for entry in self._load(context_key, embedder)
                if (entry.expires_at is None or entry.expires_at > now)
                and (oldest is None or entry.created_at > oldest)
            ]
            entries = [entry for entry in entries if len(entry.vector) == len(query)]
            if not entries:
                return None
            if _np is not None:
                matrix = _np.frombuffer(
                    b"".join(entry.vector.tobytes() for entry in entries), dtype=_np.float32
                ).reshape(len(entries), len(query))
                scores = matrix @ _np.frombuffer(query.tobytes(), dtype=_np.float32)
                best_index = int(scores.argmax())
                best_score = float(scores[best_index])
            else:
                best_index, best_score = -1, -2.0
                for index, entry in enumerate(entries):
                    score = _dot(entry.vector, query)
                    if score > best_score:
                        best_index, best_score = index, score
            if best_score < threshold:
                return None
            row = self._conn.execute(
                "SELECT value, query FROM caelum_semantic WHERE id = ?",
                (entries[best_index].row_id,),
            ).fetchone()
        if row is None:
            return None
        stored_query = row[1] if isinstance(row[1], str) else None
        return VectorMatch(
            value=json.loads(row[0]),
            score=best_score,
            query=stored_query,
            row_id=entries[best_index].row_id,
        )

    def clear(self) -> None:
        """Remove every stored embedding."""
        with self._lock:
            self._conn.execute("DELETE FROM caelum_semantic")
            self._count = 0
            self._index.clear()

    @property
    def _conn(self) -> sqlite3.Connection:
        """The connection, reopened in a forked child (SQLite forbids sharing it)."""
        if self._pid != os.getpid():
            self._finalizer.detach()
            self._pid = os.getpid()
            self._connection = sqlite3.connect(
                self.path, check_same_thread=False, isolation_level=None
            )
            self._finalizer = weakref.finalize(self, self._connection.close)
            self._index = {}
            self._init_schema()
        return self._connection

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._finalizer()
