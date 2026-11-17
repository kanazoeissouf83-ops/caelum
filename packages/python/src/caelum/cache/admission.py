"""Frequency admission: an answer enters the cache only once its question has come back.

With ``cache_admission=N`` (2 by default), an answer is stored after the API has been
called for N questions with the same meaning: the exact same request, or (with the
semantic cache) a question close enough to be served from the cache. Questions asked only
once are never stored, which keeps rare or personal requests out of the cache.

Effect on savings: with N=2 the first two requests of a question are paid and savings
start with the third one (with N=1, the default before 0.2, they start with the second).

The candidates are counted in memory, per process and per end customer (the cache key
already contains the customer), with an expiry and a maximum size so that a flood of
unique questions cannot exhaust memory.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["DEFAULT_ADMISSION", "AdmissionFilter"]

DEFAULT_ADMISSION = 2
DEFAULT_WINDOW = 24 * 60 * 60  # a candidate is forgotten after a day without repetition
DEFAULT_MAX_CANDIDATES = 10_000


@dataclass
class _Candidate:
    count: int
    expires_at: float
    vector: list[float] | None = None  # normalized embedding (semantic grouping)
    group: str = ""  # semantic context (same customer, language, model, instructions...)


def _check_n(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a whole number of at least 1")
    return value


@dataclass
class AdmissionFilter:
    """Counts how many times each question was answered by the API.

    Args:
        n: Default number of API answers needed before caching (1 = cache at once).
        customers: Per end customer overrides of ``n``.
        window: Seconds after which an unrepeated candidate is forgotten.
        max_candidates: Upper bound on remembered candidates (oldest dropped first).
    """

    n: int = DEFAULT_ADMISSION
    customers: Mapping[str, int] | None = None
    window: float = DEFAULT_WINDOW
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    clock: Callable[[], float] = time.monotonic
    _per_customer: dict[str, int] = field(default_factory=lambda: {})
    _candidates: OrderedDict[str, _Candidate] = field(default_factory=lambda: OrderedDict())
    _lock: threading.Lock = field(default_factory=threading.Lock)
    admitted: int = 0
    refused: int = 0

    def __post_init__(self) -> None:
        _check_n(self.n, "cache_admission")
        for name, value in (self.customers or {}).items():
            self.set_admission(name, value)
        if self.window <= 0 or self.max_candidates <= 0:
            raise ValueError("window and max_candidates must be positive")

    # -- settings ------------------------------------------------------------------

    def set_admission(self, customer: str, n: int) -> None:
        """Set N for one end customer."""
        if not isinstance(customer, str) or not customer:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("the customer must be a non-empty string")
        value = _check_n(n, "cache_admission")
        with self._lock:
            self._per_customer[customer] = value

    def admission_for(self, customer: str | None) -> int:
        """N in force for ``customer``."""
        if customer is None:
            return self.n
        with self._lock:
            return self._per_customer.get(customer, self.n)

    @property
    def pending(self) -> int:
        """Questions seen fewer than N times, still remembered."""
        with self._lock:
            self._expire(self.clock())
            return len(self._candidates)

    # -- counting ------------------------------------------------------------------

    def _expire(self, now: float) -> None:
        # Caller holds the lock. Candidates are kept in order of last use.
        while self._candidates:
            key, candidate = next(iter(self._candidates.items()))
            if candidate.expires_at > now:
                break
            del self._candidates[key]

    def _find_similar(self, group: str, vector: Sequence[float], threshold: float) -> str | None:
        # Caller holds the lock. Linear scan: the store is bounded by max_candidates.
        best_key, best_score = None, threshold
        for key, candidate in self._candidates.items():
            if candidate.group != group or candidate.vector is None:
                continue
            if len(candidate.vector) != len(vector):
                continue
            score = math.fsum(a * b for a, b in zip(candidate.vector, vector, strict=True))
            if score >= best_score:
                best_key, best_score = key, score
        return best_key

    def observe(
        self,
        key: str,
        customer: str | None,
        *,
        group: str = "",
        vector: Sequence[float] | None = None,
        threshold: float = 1.0,
    ) -> bool:
        """Count one API answer for this question; True when it may now be cached.

        Args:
            key: Identity of the exact question (its cache key).
            customer: End customer (for its own N).
            group: Semantic context; only candidates of the same group are grouped.
            vector: Normalized embedding, to group questions with the same meaning.
            threshold: Similarity from which two questions count as the same one.
        """
        n = self.admission_for(customer)
        if n <= 1:
            with self._lock:
                self.admitted += 1
            return True
        now = self.clock()
        with self._lock:
            self._expire(now)
            found = key if key in self._candidates else None
            if found is None and vector is not None:
                found = self._find_similar(group, vector, threshold)
            if found is None:
                self._candidates[key] = _Candidate(
                    count=1,
                    expires_at=now + self.window,
                    vector=list(vector) if vector is not None else None,
                    group=group,
                )
                while len(self._candidates) > self.max_candidates:
                    self._candidates.popitem(last=False)
                self.refused += 1
                return False
            candidate = self._candidates[found]
            candidate.count += 1
            if candidate.count >= n:
                del self._candidates[found]
                self.admitted += 1
                return True
            candidate.expires_at = now + self.window
            self._candidates.move_to_end(found)
            self.refused += 1
            return False

    def clear(self) -> None:
        """Forget every candidate."""
        with self._lock:
            self._candidates.clear()
