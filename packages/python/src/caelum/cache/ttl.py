"""Lifetime of cached answers, by default and per end customer.

Bounds: at least 1 minute, at most 30 days (the limit of the CAELUM Cloud shared memory).
``None`` (formerly "never expires") now means the maximum, 30 days.

The expiry of an entry is always ``created_at + lifetime``, locally and in CAELUM Cloud.
When a customer's lifetime is **shortened**, it also applies to the entries already stored
(an entry older than the new lifetime is no longer served): the safe choice, since a
shorter lifetime usually means "these answers go stale faster than I thought". When it
is **lengthened**, existing entries keep their original expiry (they are not extended):
only new answers get the longer lifetime.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

__all__ = ["DEFAULT_TTL", "MAX_TTL", "MIN_TTL", "TtlPolicy", "check_ttl"]

MIN_TTL = 60.0  # 1 minute
MAX_TTL = 30 * 24 * 60 * 60.0  # 30 days
DEFAULT_TTL = 24 * 60 * 60.0  # 24 hours


def check_ttl(value: Any, where: str = "cache_ttl") -> float:
    """A valid lifetime in seconds, or ValueError with a clear message (French, English)."""
    if value is None:
        return MAX_TTL
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(
            f"{where} : durée invalide ({value!r}). Donnez un nombre de secondes entre 60 "
            f"(1 minute) et 2 592 000 (30 jours), par exemple 86400 pour 24 heures. "
            f"[{where} must be a number of seconds between 60 and 2592000]"
        )
    if value <= 0:
        raise ValueError(
            f"{where} : la durée doit être positive (reçu {value!r}). Donnez un nombre de "
            f"secondes entre 60 (1 minute) et 2 592 000 (30 jours). "
            f"[{where} must be a positive number of seconds]"
        )
    if value < MIN_TTL or value > MAX_TTL:
        raise ValueError(
            f"{where} : {value!r} secondes est hors limites. La durée de vie doit être "
            f"comprise entre 60 secondes (1 minute) et 2 592 000 secondes (30 jours). "
            f"[{where} must be between 60 and 2592000 seconds]"
        )
    return float(value)


class TtlPolicy:
    """Default lifetime plus per end customer overrides (thread-safe)."""

    def __init__(
        self, default: Any = DEFAULT_TTL, customers: Mapping[str, Any] | None = None
    ) -> None:
        self.default = check_ttl(default, "cache_ttl")
        self._customers: dict[str, float] = {}
        self._lock = threading.Lock()
        if customers is not None:
            if not isinstance(customers, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    "cache_customer_ttl doit associer un nom de client à une durée en "
                    "secondes, par exemple {'acme': 3600} "
                    "[cache_customer_ttl must map a customer name to seconds]"
                )
            for name, value in customers.items():
                self.set_customer(name, value)

    def set_customer(self, customer: str, ttl: Any) -> None:
        """Set the lifetime of one end customer (applies to its stored entries too)."""
        if not isinstance(customer, str) or not customer:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("cache_customer_ttl : le nom du client doit être un texte non vide")
        value = check_ttl(ttl, f"cache_customer_ttl[{customer!r}]")
        with self._lock:
            self._customers[customer] = value

    def ttl_for(self, customer: str | None) -> float:
        """Lifetime in force for ``customer``."""
        if not customer:
            return self.default
        with self._lock:
            return self._customers.get(customer, self.default)
