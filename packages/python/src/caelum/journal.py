"""Per-request journal: what the cache did for every request, and why.

One line per request: id, time, end customer, server, outcome (``hit`` / ``gray`` /
``miss`` / ``excluded`` / ``bypassed``), similarity score, cache entry used, cost avoided
(real token cost when known, otherwise the configured estimate) and response time.

Where it lives and who can read it: a SQLite file on the machine that runs your app
(``~/.caelum/journal.sqlite`` by default, private file permissions). It is never sent to
CAELUM Cloud. Only people with access to that machine can read it.

Privacy:

* an **excluded** request (personal or time-bound data) is logged with its category only,
  never its text; a **bypassed** request (conversation, ``cache_bypass``) has no text either;
* for other requests the question asked, and for hits the stored question that was
  reused, are logged: they passed the exclusions;
* lines older than ``journal_retention_days`` (30 by default) are purged automatically,
  the file is capped at ``journal_max_rows`` lines (oldest dropped), and
  :meth:`Journal.clear` (or ``caelum journal clear``) erases it in one call.

Writing never slows a request down: lines are queued and written in batches by a
background thread. If writing fails, the request is unaffected and the error is counted.
"""

from __future__ import annotations

import atexit
import os
import queue
import sqlite3
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from caelum.cache.stores import MEMORY, caelum_home, prepare_private_file
from caelum.utils import logger

__all__ = [
    "DEFAULT_CALL_COST",
    "DEFAULT_RETENTION_DAYS",
    "OUTCOMES",
    "Journal",
    "JournalEntry",
    "default_journal_path",
]

OUTCOMES = ("hit", "gray", "miss", "excluded", "bypassed")
# Not a request: a cache clearing ("clear" or "clear_older_than"), noted for the record.
PURGE = "purge"
DEFAULT_RETENTION_DAYS = 30
DEFAULT_MAX_ROWS = 100_000
DEFAULT_CALL_COST = 0.002  # $ per avoided AI call, when the real token cost is unknown
# Estimates of the checks made before serving a semantic answer, when their real tokens
# are unknown: one gpt-4o-mini judge call (measured: ~380 tokens) and one embedding.
DEFAULT_JUDGE_CALL_COST = 0.00006
DEFAULT_EMBEDDING_CALL_COST = 0.000001
_QUEUE_SIZE = 10_000
_BATCH = 500
_FLUSH_EVERY = 0.5
_PURGE_EVERY = 600.0

_COLUMNS = (
    "request_id",
    "ts",
    "customer",
    "server",
    "outcome",
    "reason",
    "cache_kind",
    "score",
    "entry_id",
    "saved_usd",
    "saved_source",
    "latency_ms",
    "question",
    "origin_question",
    "threshold",
    "threshold_source",
    "judge_calls",
    "judge_cost_usd",
    "judge_cost_source",
    "embedding_cost_usd",
    "embedding_cost_source",
    "overhead_ms",
    "admission_counter",
    # Reserved for the shadow-mode comparisons (task 8): always NULL for now.
    "shadow_served_answer",
    "shadow_fresh_answer",
    "shadow_agreement",
    "shadow_similarity",
)


def default_journal_path() -> Path:
    return caelum_home() / "journal.sqlite"


@dataclass
class JournalEntry:
    """One request. Text fields stay None for excluded and bypassed requests."""

    outcome: str
    customer: str = ""
    server: str = ""
    reason: str | None = None  # exclusion category, or "conversation" / "cache_bypass"
    cache_kind: str | None = None  # "exact" or "semantic" for hits
    score: float | None = None
    entry_id: str | None = None
    saved_usd: float = 0.0
    saved_source: str | None = None  # "tokens" (real cost) or "estimate"
    latency_ms: float = 0.0
    question: str | None = None
    origin_question: str | None = None
    threshold: float | None = None  # similarity threshold actually used (semantic lookups)
    # "request", "request_floor", "strict", "customer" or "default"
    threshold_source: str | None = None
    # Checks made before serving a semantic answer (paid on hits AND misses).
    judge_calls: int = 0
    judge_cost_usd: float = 0.0
    judge_cost_source: str | None = None  # "tokens", "estimate" or "mixed"
    embedding_cost_usd: float = 0.0
    embedding_cost_source: str | None = None  # "tokens" or "estimate"
    overhead_ms: float = 0.0  # time spent on the embedding and the judge
    admission_counter: str | None = None  # "local" or "shared" (CAELUM Cloud), on misses
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    def row(self) -> tuple[Any, ...]:
        private = self.outcome in ("excluded", "bypassed")
        return (
            self.request_id,
            self.ts,
            self.customer,
            self.server,
            self.outcome,
            self.reason,
            self.cache_kind,
            self.score,
            self.entry_id,
            self.saved_usd,
            self.saved_source,
            self.latency_ms,
            None if private else self.question,
            None if private else self.origin_question,
            self.threshold,
            self.threshold_source,
            self.judge_calls,
            self.judge_cost_usd,
            self.judge_cost_source,
            self.embedding_cost_usd,
            self.embedding_cost_source,
            self.overhead_ms,
            self.admission_counter,
            None,
            None,
            None,
            None,
        )


def _seconds(value: datetime | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


class Journal:
    """Local per-request journal (see the module documentation).

    Args:
        path: SQLite file (default ``~/.caelum/journal.sqlite``; ``":memory:"`` for tests).
        retention_days: Lines older than this are purged automatically.
        max_rows: Upper bound on stored lines; the oldest are dropped first.
        server: Name of this server in the journal (default: the host name).
        call_cost: Estimated cost of one avoided AI call, in dollars.
        customer_call_cost: Per end customer ``call_cost``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        retention_days: float = DEFAULT_RETENTION_DAYS,
        max_rows: int = DEFAULT_MAX_ROWS,
        server: str | None = None,
        call_cost: float = DEFAULT_CALL_COST,
        judge_call_cost: float = DEFAULT_JUDGE_CALL_COST,
        embedding_call_cost: float = DEFAULT_EMBEDDING_CALL_COST,
        customer_call_cost: Mapping[str, float] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not (isinstance(retention_days, (int, float)) and retention_days > 0):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("journal_retention_days must be a positive number of days")
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("journal_max_rows must be a whole number of at least 1")
        self.call_cost = _check_cost(call_cost, "journal_call_cost")
        self.judge_call_cost = _check_cost(judge_call_cost, "judge_call_cost")
        self.embedding_call_cost = _check_cost(embedding_call_cost, "embedding_call_cost")
        self._costs = {
            str(name): _check_cost(value, f"journal_customer_call_cost[{name!r}]")
            for name, value in (customer_call_cost or {}).items()
        }
        self.path = str(path) if path is not None else str(default_journal_path())
        self.retention_days = float(retention_days)
        self.max_rows = max_rows
        self.server = server if server is not None else _host()
        self._clock = clock
        self._queue: queue.Queue[JournalEntry] = queue.Queue(maxsize=_QUEUE_SIZE)
        self._lock = threading.Lock()  # one writer at a time (thread, flush(), clear())
        self._wakeup = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._last_purge = 0.0
        self.errors = 0  # failed writes (the requests themselves were unaffected)
        self.dropped = 0  # lines dropped because the queue was full
        if self.path != MEMORY:
            prepare_private_file(self.path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        try:
            self._init_schema()
        except BaseException:
            self._conn.close()
            raise
        self._finalizer = weakref.finalize(self, self._conn.close)
        _journals.add(self)

    def _init_schema(self) -> None:
        with self._lock:
            if self.path != MEMORY:
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS caelum_journal ("
                " request_id TEXT PRIMARY KEY, ts REAL NOT NULL, customer TEXT NOT NULL,"
                " server TEXT NOT NULL, outcome TEXT NOT NULL, reason TEXT, cache_kind TEXT,"
                " score REAL, entry_id TEXT, saved_usd REAL NOT NULL, saved_source TEXT,"
                " latency_ms REAL NOT NULL, question TEXT, origin_question TEXT,"
                " shadow_served_answer TEXT, shadow_fresh_answer TEXT, shadow_agreement INTEGER,"
                " shadow_similarity REAL)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS caelum_journal_ts ON caelum_journal (customer, ts)"
            )
            columns = {r[1] for r in self._conn.execute("PRAGMA table_info(caelum_journal)")}
            for name, kind in (
                ("threshold", "REAL"),
                ("threshold_source", "TEXT"),
                ("judge_calls", "INTEGER"),
                ("judge_cost_usd", "REAL"),
                ("judge_cost_source", "TEXT"),
                ("embedding_cost_usd", "REAL"),
                ("embedding_cost_source", "TEXT"),
                ("overhead_ms", "REAL"),
                ("admission_counter", "TEXT"),
            ):
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE caelum_journal ADD COLUMN {name} {kind}")

    # -- costs ---------------------------------------------------------------------

    def call_cost_for(self, customer: str | None) -> float:
        return self._costs.get(customer or "", self.call_cost)

    # -- writing ---------------------------------------------------------------------

    def record(self, entry: JournalEntry) -> None:
        """Queue one line. Never blocks and never raises."""
        if self._closed:
            return
        if not entry.server:
            entry.server = self.server
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self.dropped += 1
            return
        self._start()
        if self._queue.qsize() >= _BATCH:
            self._wakeup.set()

    def _start(self) -> None:
        if self._thread is None:
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name="caelum-journal", daemon=True
                    )
                    self._thread.start()

    def _run(self) -> None:
        while not self._closed:
            self._wakeup.wait(_FLUSH_EVERY)
            self._wakeup.clear()
            self.flush()

    def _write(self, rows: list[tuple[Any, ...]]) -> None:
        """Insert a batch (separate method so tests can simulate a broken disk)."""
        self._conn.executemany(
            f"INSERT OR REPLACE INTO caelum_journal ({', '.join(_COLUMNS)})"
            f" VALUES ({', '.join('?' * len(_COLUMNS))})",
            rows,
        )

    def flush(self) -> None:
        """Write the queued lines now (done every half second in the background)."""
        rows: list[tuple[Any, ...]] = []
        while True:
            try:
                rows.append(self._queue.get_nowait().row())
            except queue.Empty:
                break
        with self._lock:
            if self._finalizer.alive is False:
                return  # closed: the file is gone (late wake-up of the writer thread)
            try:
                if rows:
                    self._write(rows)
                self._maintain()
            except Exception:
                self.errors += 1
                logger.warning("caelum: could not write the request journal", exc_info=True)

    def _maintain(self) -> None:
        # Caller holds the lock: retention purge and size cap, at most every 10 minutes.
        now = self._clock()
        if now - self._last_purge < _PURGE_EVERY:
            return
        self._last_purge = now
        self._purge(now)

    def _purge(self, now: float) -> None:
        self._conn.execute(
            "DELETE FROM caelum_journal WHERE ts < ?", (now - self.retention_days * 86_400,)
        )
        (count,) = self._conn.execute("SELECT COUNT(*) FROM caelum_journal").fetchone()
        if count > self.max_rows:
            self._conn.execute(
                "DELETE FROM caelum_journal WHERE request_id IN (SELECT request_id FROM"
                " caelum_journal ORDER BY ts LIMIT ?)",
                (count - self.max_rows,),
            )

    def purge(self) -> None:
        """Apply the retention period and the size cap now."""
        self.flush()
        with self._lock:
            self._purge(self._clock())

    def clear(self, customer: str | None = None) -> int:
        """Erase the journal (only ``customer``'s lines when given). Returns lines erased."""
        self.flush()
        with self._lock:
            if customer is None:
                cursor = self._conn.execute("DELETE FROM caelum_journal")
            else:
                cursor = self._conn.execute(
                    "DELETE FROM caelum_journal WHERE customer = ?", (customer,)
                )
            return cursor.rowcount

    # -- reading ---------------------------------------------------------------------

    def entries(self, customer: str | None = None) -> list[dict[str, Any]]:
        """Every stored line (of ``customer`` only when given), oldest first."""
        self.flush()
        sql = f"SELECT {', '.join(_COLUMNS)} FROM caelum_journal"
        args: tuple[Any, ...] = ()
        if customer is not None:
            sql, args = sql + " WHERE customer = ?", (customer,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY ts", args).fetchall()
        return [dict(zip(_COLUMNS, row, strict=True)) for row in rows]

    def stats(
        self,
        customer: str | None,
        *,
        since: datetime | float | None = None,
        until: datetime | float | None = None,
        lowest: int = 20,
    ) -> dict[str, Any]:
        """Statistics of one end customer over a period (``customer=None``: the requests
        made outside any ``caelum.customer()`` block).

        Returns the number of requests, the real hit rate, the count per outcome, the
        estimated savings, and the ``lowest`` hits with the lowest similarity score (the
        riskiest to check), with the question asked and the stored question reused.
        """
        self.flush()
        where = ["customer = ?"]
        args: list[Any] = [customer or ""]
        start, end = _seconds(since), _seconds(until)
        if start is not None:
            where.append("ts >= ?")
            args.append(start)
        if end is not None:
            where.append("ts < ?")
            args.append(end)
        clause = " AND ".join(where)
        with self._lock:
            by_outcome = dict(
                self._conn.execute(
                    f"SELECT outcome, COUNT(*) FROM caelum_journal WHERE {clause}"
                    " AND outcome != 'purge' GROUP BY outcome",
                    args,
                ).fetchall()
            )
            saved, estimated = self._conn.execute(
                "SELECT COALESCE(SUM(saved_usd), 0),"
                " COALESCE(SUM(CASE WHEN saved_source = 'estimate' THEN saved_usd END), 0)"
                f" FROM caelum_journal WHERE {clause}",
                args,
            ).fetchone()
            judge, embeddings, judge_calls = self._conn.execute(
                "SELECT COALESCE(SUM(judge_cost_usd), 0), COALESCE(SUM(embedding_cost_usd), 0),"
                " COALESCE(SUM(judge_calls), 0)"
                f" FROM caelum_journal WHERE {clause} AND outcome != 'purge'",
                args,
            ).fetchone()
            hit_ms, overhead_ms = self._conn.execute(
                "SELECT AVG(latency_ms), AVG(overhead_ms)"
                f" FROM caelum_journal WHERE {clause} AND outcome = 'hit'",
                args,
            ).fetchone()
            (ai_ms,) = self._conn.execute(
                "SELECT AVG(latency_ms) FROM caelum_journal"
                f" WHERE {clause} AND outcome IN ('miss', 'gray', 'excluded', 'bypassed')",
                args,
            ).fetchone()
            riskiest = self._conn.execute(
                "SELECT request_id, ts, score, question, origin_question, entry_id"
                f" FROM caelum_journal WHERE {clause} AND outcome = 'hit' AND score IS NOT NULL"
                " ORDER BY score ASC, ts DESC LIMIT ?",
                [*args, lowest],
            ).fetchall()
        requests = sum(by_outcome.values())
        hits = by_outcome.get("hit", 0)
        hit_rate = hits / requests if requests else 0.0
        checks = judge + embeddings
        net = saved - checks
        # Break-even: net >= 0 when hit_rate * (saving per hit) >= (checks per request).
        per_hit = saved / hits if hits else self.call_cost_for(customer)
        per_request = checks / requests if requests else 0.0
        break_even = None if per_hit <= 0 or not requests else min(per_request / per_hit, 1.0)
        alerts = _alerts(
            requests, hits, hit_rate, net, checks, break_even, hit_ms, ai_ms, overhead_ms
        )
        return {
            "customer": customer or "",
            "requests": requests,
            "hits": hits,
            "hit_rate": hits / requests if requests else 0.0,
            "by_outcome": {name: by_outcome.get(name, 0) for name in OUTCOMES},
            "saved_usd": saved,  # gross savings (kept for compatibility)
            "saved_usd_estimated": estimated,  # part based on the per-call estimate
            "gross_saved_usd": saved,
            "judge_cost_usd": judge,
            "judge_calls": judge_calls,
            "embedding_cost_usd": embeddings,
            "verification_cost_usd": checks,
            "net_saved_usd": net,
            "avg_hit_ms": hit_ms,
            "avg_ai_ms": ai_ms,
            "avg_check_ms": overhead_ms,
            "break_even_hit_rate": break_even,
            "alerts": alerts,
            "lowest_score_hits": [
                dict(
                    zip(
                        ("request_id", "ts", "score", "question", "origin_question", "entry_id"),
                        r,
                        strict=True,
                    )
                )
                for r in riskiest
            ],
            "journal_errors": self.errors,
        }

    def close(self) -> None:
        """Write what is queued and close the file."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
            self._wakeup.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)
            with self._lock:  # never close the file under a write in progress
                self._finalizer()


def _alerts(
    requests: int,
    hits: int,
    hit_rate: float,
    net: float,
    checks: float,
    break_even: float | None,
    hit_ms: float | None,
    ai_ms: float | None,
    check_ms: float | None,
) -> list[str]:
    """Plain-French warnings when the cache costs more than it saves, or is slower."""
    alerts: list[str] = []
    if requests and net < 0:
        reason = "trop peu de questions répétées"
        if break_even is not None and hit_rate < break_even:
            reason += (
                f" : {100 * hit_rate:.1f} % des questions sont servies par le cache, il en "
                f"faudrait au moins {100 * break_even:.1f} % pour être rentable"
            )
        elif hits == 0:
            reason += " : aucune question n'a encore été servie par le cache"
        alerts.append(
            f"Économie nette négative ({net:+.6f} $) : les vérifications (juge et embeddings, "
            f"{checks:.6f} $) coûtent plus que les appels à l'IA évités. Cause probable : "
            f"{reason}. Pistes : laisser le cache tourner plus longtemps, désactiver le cache "
            "intelligent pour ce client, ou utiliser un juge moins cher."
        )
    if hit_ms is not None and ai_ms is not None and hit_ms > ai_ms:
        detail = f" dont {check_ms:.0f} ms de vérifications" if check_ms else ""
        alerts.append(
            f"Le cache est plus lent que l'IA directe pour ce client : {hit_ms:.0f} ms en "
            f"moyenne par réponse servie par le cache{detail}, contre {ai_ms:.0f} ms pour un "
            "appel direct à l'IA. Cause probable : le juge ajoute un aller-retour vers l'IA "
            "à chaque réponse servie. Pistes : un juge plus rapide, ou le cache exact seul."
        )
    return alerts


def _check_cost(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{what} must be a number of dollars, 0 or more")
    return float(value)


def _host() -> str:
    try:
        import socket

        return socket.gethostname() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


_journals: weakref.WeakSet[Journal] = weakref.WeakSet()


@atexit.register
def _close_all() -> None:  # pragma: no cover - runs at interpreter shutdown
    for journal in list(_journals):
        try:
            journal.close()
        except Exception:
            pass


def clear_journal_file(
    path: str | os.PathLike[str] | None = None, customer: str | None = None
) -> int:
    """Erase a journal file without creating a client (used by ``caelum journal clear``)."""
    location = Path(path) if path is not None else default_journal_path()
    if not location.exists():
        return 0
    with closing(sqlite3.connect(location)) as conn, conn:
        if customer is None:
            return conn.execute("DELETE FROM caelum_journal").rowcount
        return conn.execute("DELETE FROM caelum_journal WHERE customer = ?", (customer,)).rowcount
