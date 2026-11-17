"""In-memory metrics: what CAELUM processed, what it saved, how fast.

Every client exposes ``client.metrics``::

    client.metrics.print_summary()

    CAELUM Local Metrics
    Requests:        1 240
    Cache hits:        743 (60%)
    Cost saved:      $ 24.30
    Avg latency:      180ms

Metrics live in memory only: nothing is sent anywhere. Costs are *estimates* based on
:mod:`caelum.pricing`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Literal, TextIO

from caelum.utils import format_usd

__all__ = ["Metrics"]

CacheKind = Literal["exact", "semantic"]


def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


class Metrics:
    """Thread-safe counters for one client (shared by its ``copy()`` / ``with_options()``).

    Args:
        max_latency_samples: Number of most recent latencies kept for percentiles.
    """

    def __init__(self, max_latency_samples: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._max_samples = max_latency_samples
        self.reset()

    # -- recording (used by the CAELUM engine) -----------------------------------

    def reset(self) -> None:
        """Reset every counter to zero."""
        with self._lock:
            self.total_requests = 0
            self.cache_hits_exact = 0
            self.cache_hits_semantic = 0
            self.cache_misses = 0
            self.failed_requests = 0
            self.total_retries = 0
            self.total_fallbacks_triggered = 0
            self.routed_requests = 0
            self.unpriced_requests = 0
            self.total_cost_estimated_usd = 0.0
            self.cost_saved_estimated_usd = 0.0
            self.embedding_cost_estimated_usd = 0.0
            self.semantic_verifications = 0
            self.semantic_rejections = 0
            self.semantic_refreshes = 0
            self.exclusions: dict[str, int] = {}
            self.shared_admitted = 0  # answers admitted by the shared (Cloud) counter
            self.shared_waiting = 0  # counted in the Cloud, not repeated enough yet
            self.shared_unavailable = 0  # Cloud unreachable: the local counter was used
            self.verifier_cost_estimated_usd = 0.0
            self.total_latency_ms = 0.0
            self.latencies: deque[float] = deque(maxlen=self._max_samples)
            self.requests_by_model: dict[str, int] = {}

    def _count_request(self, model: str | None, latency_ms: float) -> None:
        # Caller holds the lock.
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)
        key = model or "unknown"
        self.requests_by_model[key] = self.requests_by_model.get(key, 0) + 1

    def record_api_call(
        self, *, model: str | None, latency_ms: float, cost_usd: float | None, cache_miss: bool
    ) -> None:
        """Record a request answered by the provider API."""
        with self._lock:
            self._count_request(model, latency_ms)
            if cache_miss:
                self.cache_misses += 1
            if cost_usd is None:
                self.unpriced_requests += 1
            else:
                self.total_cost_estimated_usd += cost_usd

    def record_cache_hit(
        self, *, kind: CacheKind, model: str | None, latency_ms: float, saved_usd: float | None
    ) -> None:
        """Record a request answered from a CAELUM cache."""
        with self._lock:
            self._count_request(model, latency_ms)
            if kind == "exact":
                self.cache_hits_exact += 1
            else:
                self.cache_hits_semantic += 1
            if saved_usd is not None:
                self.cost_saved_estimated_usd += saved_usd

    def record_failure(self, *, model: str | None, latency_ms: float) -> None:
        """Record a request that raised an error."""
        with self._lock:
            self._count_request(model, latency_ms)
            self.failed_requests += 1

    def record_retry(self) -> None:
        """Record one retry attempt."""
        with self._lock:
            self.total_retries += 1

    def record_route(self) -> None:
        """Record one request sent to another model by the router."""
        with self._lock:
            self.routed_requests += 1

    def record_fallback(self) -> None:
        """Record one switch to a fallback provider/model."""
        with self._lock:
            self.total_fallbacks_triggered += 1

    def record_embedding_cost(self, cost_usd: float | None) -> None:
        """Record the cost of an embedding computed for the semantic cache."""
        if cost_usd is None:
            return
        with self._lock:
            self.embedding_cost_estimated_usd += cost_usd
            self.total_cost_estimated_usd += cost_usd

    def record_verification(self, *, accepted: bool, cost_usd: float | None) -> None:
        """Record one semantic-cache verification by the LLM judge."""
        with self._lock:
            self.semantic_verifications += 1
            if not accepted:
                self.semantic_rejections += 1
            if cost_usd is not None:
                self.verifier_cost_estimated_usd += cost_usd
                self.total_cost_estimated_usd += cost_usd

    def attach_admission(self, admission: Any) -> None:
        """Report the frequency-admission counters of this client in :meth:`summary`."""
        self._admission = admission

    def attach_journal(self, journal: Any) -> None:
        """Report the request journal's write errors in :meth:`summary`."""
        self._journal = journal

    def record_shared_admission(self, admitted: bool | None) -> None:
        """Record the answer of the shared admission counter (None = Cloud unreachable)."""
        with self._lock:
            if admitted is None:
                self.shared_unavailable += 1
            elif admitted:
                self.shared_admitted += 1
            else:
                self.shared_waiting += 1

    def record_exclusion(self, category: str) -> None:
        """Record one request kept out of the cache, and why (never its text)."""
        with self._lock:
            self.exclusions[category] = self.exclusions.get(category, 0) + 1

    def record_semantic_refresh(self) -> None:
        """Record one gray-zone match: the API was called and the entry replaced."""
        with self._lock:
            self.semantic_refreshes += 1

    # -- reading -------------------------------------------------------------------

    @property
    def cache_hits(self) -> int:
        """Total cache hits (exact + semantic)."""
        return self.cache_hits_exact + self.cache_hits_semantic

    def summary(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot of all metrics."""
        admission = getattr(self, "_admission", None)
        journal = getattr(self, "_journal", None)
        with self._lock:
            ordered = sorted(self.latencies)
            hits = self.cache_hits_exact + self.cache_hits_semantic
            requests = self.total_requests
            return {
                "requests": requests,
                "cache_hits": hits,
                "cache_hits_exact": self.cache_hits_exact,
                "cache_hits_semantic": self.cache_hits_semantic,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": hits / requests if requests else 0.0,
                "api_calls": requests - hits - self.failed_requests,
                "failed_requests": self.failed_requests,
                "retries": self.total_retries,
                "fallbacks": self.total_fallbacks_triggered,
                "routed": self.routed_requests,
                "cost_usd": self.total_cost_estimated_usd,
                "cost_saved_usd": self.cost_saved_estimated_usd,
                "embedding_cost_usd": self.embedding_cost_estimated_usd,
                "semantic_verifications": self.semantic_verifications,
                "semantic_rejections": self.semantic_rejections,
                "semantic_refreshes": self.semantic_refreshes,
                "excluded_requests": sum(self.exclusions.values()),
                "journal_errors": journal.errors if journal is not None else 0,
                "exclusions": dict(self.exclusions),
                "admission_pending": admission.pending if admission is not None else 0,
                "admission_admitted": admission.admitted if admission is not None else 0,
                "admission_refused": admission.refused if admission is not None else 0,
                "admission_shared_admitted": self.shared_admitted,
                "admission_shared_waiting": self.shared_waiting,
                "admission_shared_unavailable": self.shared_unavailable,
                "verifier_cost_usd": self.verifier_cost_estimated_usd,
                "unpriced_requests": self.unpriced_requests,
                "latency_avg_ms": self.total_latency_ms / requests if requests else 0.0,
                "latency_p50_ms": _percentile(ordered, 50),
                "latency_p95_ms": _percentile(ordered, 95),
                "requests_by_model": dict(self.requests_by_model),
            }

    def format_summary(self) -> str:
        """Return the human-readable summary as plain text."""
        s = self.summary()
        width = max(len(_thousands(s["requests"])), 5)
        lines = [
            "CAELUM Local Metrics",
            f"Requests:      {_thousands(s['requests']):>{width + 2}}",
            f"Cache hits:    {_thousands(s['cache_hits']):>{width + 2}} "
            f"({s['cache_hit_rate']:.0%})",
            f"Cost saved:    {format_usd(s['cost_saved_usd'])}",
            f"Cost spent:    {format_usd(s['cost_usd'])}",
            f"Avg latency:   {s['latency_avg_ms']:>{width}.0f}ms",
            f"p95 latency:   {s['latency_p95_ms']:>{width}.0f}ms",
        ]
        if s["retries"]:
            lines.append(f"Retries:       {_thousands(s['retries']):>{width + 2}}")
        if s["failed_requests"]:
            lines.append(f"Failed:        {_thousands(s['failed_requests']):>{width + 2}}")
        if s["unpriced_requests"]:
            lines.append(
                f"(cost unknown for {s['unpriced_requests']} request(s): model not priced)"
            )
        return "\n".join(lines)

    def print_summary(self, file: TextIO | None = None) -> None:
        """Print the summary (pretty table if the ``rich`` extra is installed)."""
        target = file if file is not None else sys.stdout
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            print(self.format_summary(), file=target)
            return
        s = self.summary()
        table = Table(title="CAELUM Local Metrics", show_header=False)
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("Requests", _thousands(s["requests"]))
        table.add_row("Cache hits", f"{_thousands(s['cache_hits'])} ({s['cache_hit_rate']:.0%})")
        table.add_row("Cost saved", format_usd(s["cost_saved_usd"]))
        table.add_row("Cost spent", format_usd(s["cost_usd"]))
        table.add_row("Avg latency", f"{s['latency_avg_ms']:.0f}ms")
        table.add_row("p95 latency", f"{s['latency_p95_ms']:.0f}ms")
        if s["retries"]:
            table.add_row("Retries", _thousands(s["retries"]))
        if s["failed_requests"]:
            table.add_row("Failed", _thousands(s["failed_requests"]))
        Console(file=target).print(table)
        if s["unpriced_requests"]:
            print(
                f"(cost unknown for {s['unpriced_requests']} request(s): model not priced)",
                file=target,
            )

    def export_json(self, path: str | os.PathLike[str]) -> None:
        """Write the summary to a JSON file (missing folders are created)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def __repr__(self) -> str:
        s = self.summary()
        return f"<caelum.Metrics requests={s['requests']} cache_hits={s['cache_hits']}>"
