"""Automatic cache exclusions: questions about one person or one moment go to the AI.

A question that contains an order or tracking number, a phone number, an email address,
an amount of money, a relative or precise date or time, payment / identity data, or a
secret the user pasted (an API key, access token or private key) is **never** read from
the cache, **never** written to it, and never counted for frequency admission: its text is
kept nowhere by CAELUM. It goes straight to the AI.

WHY: such questions have an answer that belongs to one customer or one moment ("Where is
order 4471?", "Can you deliver before 5pm tomorrow?"). Reusing it for someone else would
be wrong, and storing it would keep personal data.

The detection is tuned for West African customers writing in French or English (+226,
+233, +225 numbers, FCFA / XOF / GHS amounts). It only looks at the **user's question**,
never at the system instructions. It does not try to detect city or product names: "Do you
deliver to Bobo-Dioulasso?" stays cacheable.

Every category can be switched off, globally or per end customer, and custom regular
expressions can be added.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

__all__ = ["CATEGORIES", "ExclusionPolicy", "ExclusionRules"]

_I = re.IGNORECASE | re.UNICODE

# category -> patterns. Each pattern needs a digit or a very specific word, so that
# ordinary FAQ questions are never excluded.
_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "order": (
        # "commande n° 4471", "colis AB123456", "order #4471", "facture 2026-118"
        re.compile(
            r"\b(?:commandes?|cdes?|orders?|colis|suivi|tracking|factures?|invoices?|"
            r"tickets?|r[ée]f(?:[ée]rences?)?|dossiers?|parcels?|packages?|shipments?)"
            r"\s*(?:n[°ºo]\.?|num(?:[ée]ro)?\.?|no\.?|number|#|:)?\s*"
            r"[A-Z]{0,4}[-\s]?\d[A-Z0-9-]{2,}",
            _I,
        ),
        re.compile(r"#\s?\d{2,}"),  # "#12", "#4471"
        re.compile(r"\bn[°º]\s?\d{2,}", _I),  # "n° 12"
        re.compile(r"\b[A-Z]{2,4}\d{6,}[A-Z]{0,3}\b"),  # tracking codes: "AB123456789FR"
        # A reference word shortly followed by a 2+ digit number: "statut de 4471",
        # "commande 12", "ref: 88", "dossier numéro 7". Catches short numbers the patterns
        # above (which need 3+ digits together) would miss.
        re.compile(
            r"\b(?:commandes?|cdes?|orders?|colis|suivi|tracking|factures?|invoices?|"
            r"tickets?|r[ée]f(?:[ée]rences?)?|dossiers?|parcels?|packages?|shipments?|"
            r"statuts?|status|state)\b[^\n]{0,15}?\b\d{2,}\b",
            _I,
        ),
    ),
    "phone": (
        re.compile(
            r"(?<![\w+])\+\d{1,3}[\s.-]?\d{1,4}(?:[\s.-]?\d{2,4}){2,5}\b"
        ),  # +226 70 12 34 56
        re.compile(r"\b00\d{3}[\s.-]?\d{2}(?:[\s.-]?\d{2}){3,4}\b"),  # 00226 70 12 34 56
        re.compile(
            r"(?<![\d,.])\d{2}(?:[\s.-]?\d{2}){3,4}(?![\d,.]|\s?(?:f\b|fcfa|cfa|xof|francs?|ghs|\$|€))",
            _I,
        ),
        re.compile(r"(?<![\d,.])0\d{2}[\s.-]?\d{3}[\s.-]?\d{4}(?![\d,.])"),  # Ghana: 024 412 3456
    ),
    "email": (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", _I),),
    "amount": (
        re.compile(
            r"\d[\d\s.,]*\s?(?:f\s?cfa|cfa|xof|ghs|gh₵|ghc|c[ée]dis?|naira|ngn|usd|eur|euros?|"
            r"dollars?|francs?|f\b|\$|€|£|₦|₵)",
            _I,
        ),
        re.compile(r"(?:\$|€|£|₦|₵|\b(?:ghs|xof|fcfa|usd|eur|ngn))\s?\d", _I),
    ),
    "datetime": (
        re.compile(
            r"\b(?:aujourd'hui|aujourd’hui|demain|apr[èe]s-demain|hier|avant-hier|ce\s+soir|"
            r"ce\s+matin|cet\s+apr[èe]s-midi|cette\s+nuit|cette\s+semaine|ce\s+week-?end|"
            r"ce\s+mois(?:-ci)?|cette\s+ann[ée]e|tout\s+[àa]\s+l'heure|tout\s+de\s+suite|"
            r"today|tomorrow|yesterday|tonight|this\s+(?:morning|afternoon|evening|week|"
            r"weekend|month|year)|right\s+now)\b",
            _I,
        ),
        re.compile(
            r"\b(?:(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|semaine|mois|"
            r"week-?end|ann[ée]e)\s+(?:prochaine?|derni(?:er|ère)|pass[ée]e?)|"
            r"(?:next|last)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"week|weekend|month|year))\b",
            _I,
        ),
        re.compile(
            r"\b(?:le|du|au|jusqu'au|avant\s+le|apr[èe]s\s+le|on\s+the)\s+\d{1,2}(?:er|st|nd|rd|th)?\b",
            _I,
        ),
        re.compile(
            r"\b\d{1,2}(?:er)?\s+(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[ûu]t|"
            r"septembre|octobre|novembre|d[ée]cembre|january|february|march|april|may|june|"
            r"july|august|september|october|november|december)\b",
            _I,
        ),
        re.compile(r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b"),  # 12/11, 12-11-2026
        # A time of day ("avant 17h", "à 9 h", "17h30"), not a duration ("en 48h", "24h/24").
        re.compile(
            r"\b(?:[àa]|avant|apr[èe]s|vers|d[èe]s|jusqu'?[àa]|at|before|after|by|until)\s+"
            r"\d{1,2}\s?h(?:\s?\d{2})?\b",
            _I,
        ),
        re.compile(r"\b\d{1,2}\s?h\s?\d{2}\b", _I),
        re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", _I),
    ),
    "payment_identity": (
        # Card numbers: 13 to 19 digits, never starting with 0 (unlike "00226..." phones).
        re.compile(r"(?<![\d+])[1-9](?:[ -]?\d){12,18}(?!\d)"),
        re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b"),  # IBAN
        re.compile(r"\biban\b\s*:?\s*[A-Z]{2}\d", _I),
        re.compile(
            r"\b(?:mot\s+de\s+passe|password|passcode|mdp|code\s+pin|pin|cvv|cvc|"
            r"code\s+secret|otp|code\s+de\s+v[ée]rification|verification\s+code)"
            r"\s*(?:est|is|=|:|c'est)\s*\S+",
            _I,
        ),
        re.compile(
            r"\b(?:cnib|passeport|passport|pi[èe]ce\s+d'identit[ée]|carte\s+d'identit[ée]|"
            r"carte\s+nationale|id\s+card|identity\s+card|nin|num[ée]ro\s+d'identit[ée]|"
            r"ghana\s+card)\b[^\n]{0,25}?[A-Z]{0,4}[-\s]?\d[A-Z0-9-]{3,}",
            _I,
        ),
        # Secrets a user may paste: API keys, access tokens, private keys. Letters mean the
        # card pattern above misses them; left in the local cache or journal they would be a
        # retained secret. "sk-...", "sk-proj-...", GitHub/AWS/Slack tokens, bearer, JWT, PEM.
        re.compile(r"\b(?:sk|pk|rk)-(?:proj-)?[A-Za-z0-9]{16,}"),  # sk-..., sk-proj-..., pk-...
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
        re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),  # AWS access key id
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}", _I),  # Slack tokens
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*", _I),  # Authorization: Bearer
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}"),  # JWT
        re.compile(r"-----BEGIN(?:\s+[A-Z]+)*\s+PRIVATE KEY-----"),  # PEM private key
        # "api_key = ...", "secret: ...", "token=...", "client_secret ...".
        re.compile(
            r"\b(?:api[_-]?keys?|secrets?|access[_-]?tokens?|client[_-]?secrets?|"
            r"cl[ée]\s+(?:api|secr[èe]te)|jeton\s+d'acc[èe]s)\b\s*(?:est|is|=|:)\s*\S{6,}",
            _I,
        ),
    ),
}

# Checked in this order (the most specific first), so that a card or an ID number is
# reported as "payment_identity" rather than as a phone number.
CATEGORIES: tuple[str, ...] = ("payment_identity", "email", "order", "phone", "amount", "datetime")
assert set(CATEGORIES) == set(_PATTERNS)
MAX_PATTERN_LENGTH = 500
MAX_SCANNED_CHARS = 20_000


def _compile(patterns: object, where: str) -> tuple[re.Pattern[str], ...]:
    if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Iterable):
        raise TypeError(f"{where} must be a list of regular expressions (strings)")
    compiled: list[re.Pattern[str]] = []
    for pattern in cast("Iterable[Any]", patterns):
        if not isinstance(pattern, str) or not pattern:
            raise TypeError(f"{where}: every pattern must be a non-empty string")
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise ValueError(f"{where}: pattern longer than {MAX_PATTERN_LENGTH} characters")
        try:
            compiled.append(re.compile(pattern, _I))
        except re.error as error:
            raise ValueError(
                f"{where}: invalid regular expression {pattern!r} ({error}). "
                r"Example of a valid pattern: r'\bclient\s+\d+'"
            ) from None
    return tuple(compiled)


@dataclass(frozen=True)
class ExclusionRules:
    """Enabled categories and custom patterns for one end customer (or the default)."""

    categories: frozenset[str]
    patterns: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def build(
        cls,
        switches: Mapping[str, Any] | None = None,
        patterns: Iterable[Any] | None = None,
        *,
        base: ExclusionRules | None = None,
        where: str = "cache_exclusions",
    ) -> ExclusionRules:
        """Rules from category switches (``{"amount": False}``) and custom patterns."""
        enabled = set(base.categories if base is not None else CATEGORIES)
        if switches is not None:
            if not isinstance(switches, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"{where} must map a category name to True or False")
            for name, on in switches.items():
                if name not in CATEGORIES:
                    raise ValueError(
                        f"{where}: unknown category {name!r}; valid ones: {', '.join(CATEGORIES)}"
                    )
                if not isinstance(on, bool):
                    raise TypeError(f"{where}: {name!r} must be True or False")
                (enabled.add if on else enabled.discard)(name)
        compiled = base.patterns if base is not None else ()
        if patterns is not None:
            compiled = _compile(patterns, f"{where} patterns")
        return cls(categories=frozenset(enabled), patterns=compiled)

    def match(self, text: str) -> str | None:
        """The first matching category (``"custom"`` for a custom pattern), or None."""
        text = text[:MAX_SCANNED_CHARS]
        for category in CATEGORIES:
            if category in self.categories and any(p.search(text) for p in _PATTERNS[category]):
                return category
        for pattern in self.patterns:
            if pattern.search(text):
                return "custom"
        return None


class ExclusionPolicy:
    """Default exclusion rules plus per end customer overrides (thread-safe).

    Args:
        categories: Category switches, e.g. ``{"amount": False}`` (all on by default).
        patterns: Extra regular expressions that exclude a question.
        customers: Per end customer settings: ``{"acme": {"amount": False,
            "patterns": [r"\\bclient\\s+\\d+"]}}``. Unset categories follow the default.
    """

    def __init__(
        self,
        categories: Mapping[str, Any] | None = None,
        patterns: Iterable[Any] | None = None,
        customers: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.default = ExclusionRules.build(categories, patterns)
        self._customers: dict[str, ExclusionRules] = {}
        self._lock = threading.Lock()
        if customers is not None:
            if not isinstance(customers, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("cache_customer_exclusions must map a customer name to settings")
            for name, settings in customers.items():
                self.set_customer(name, settings)

    def set_customer(self, customer: str, settings: Mapping[str, Any]) -> None:
        """Override the rules of one end customer."""
        if not isinstance(customer, str) or not customer:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("cache_customer_exclusions: the customer must be a non-empty string")
        if not isinstance(settings, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                "cache_customer_exclusions: settings must be a mapping such as "
                "{'amount': False, 'patterns': [...]}"
            )
        switches = {k: v for k, v in settings.items() if k != "patterns"}
        rules = ExclusionRules.build(
            switches,
            settings.get("patterns"),
            base=self.default,
            where=f"cache_customer_exclusions[{customer!r}]",
        )
        with self._lock:
            self._customers[customer] = rules

    def rules_for(self, customer: str | None) -> ExclusionRules:
        if customer is None:
            return self.default
        with self._lock:
            return self._customers.get(customer, self.default)

    def check(self, text: str, customer: str | None) -> str | None:
        """Category that excludes ``text`` for ``customer``, or None if it may be cached.

        Never raises: if a pattern fails at runtime, the question is excluded (safe side).
        """
        if not text:
            return None
        try:
            return self.rules_for(customer).match(text)
        except Exception:  # pragma: no cover - defensive: never break the cache
            return "custom"
