"""What a cached answer depends on, beyond the question itself.

Two requests may only share a cached answer when they have the same:

* **end customer** (:func:`caelum.customer`): one customer's answers are never served to
  another one;
* **language** (declared with :func:`language`, otherwise detected from the question):
  an answer is only reused in the language it was written in;
* **AI model** actually called (after routing);
* **system instructions** (a fingerprint of the system / developer prompt).

These four values are hashed into every cache key, exact and semantic.

Requests that **depend on a conversation** (an earlier answer of the AI, or more than one
user message) are never read from nor written to the cache: "and to Ouaga?" means
nothing without "Do you deliver to Bobo?". :func:`cache_bypass` (or the ``cache_bypass=True``
request argument) forces the same behaviour for any other request.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
import unicodedata
from collections.abc import Generator, Mapping
from typing import Any, cast

from caelum.cloud import current_customer
from caelum.utils import UnhashableRequestError, stable_hash, to_canonical

__all__ = [
    "cache_bypass",
    "cache_bypassed",
    "context_fingerprint",
    "current_language",
    "depends_on_conversation",
    "detect_language",
    "language",
    "last_user_text",
]

_language: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "caelum_language", default=None
)
_bypass: contextvars.ContextVar[bool] = contextvars.ContextVar("caelum_cache_bypass", default=False)

_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


@contextlib.contextmanager
def language(code: str) -> Generator[None]:
    """Declare the language of the requests made inside the block (e.g. ``"fr"``).

    Overrides the automatic detection. Cached answers are only reused for requests in the
    same language::

        with caelum.language("fr"):
            client.chat.completions.create(...)
    """
    if not isinstance(code, str) or not _LANGUAGE_CODE.match(code.strip()):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError('language must be a code such as "fr", "en" or "pt-BR"')
    token = _language.set(code.strip().lower())
    try:
        yield
    finally:
        _language.reset(token)


def current_language() -> str | None:
    """Language declared with :func:`language`, if any."""
    return _language.get()


@contextlib.contextmanager
def cache_bypass() -> Generator[None]:
    """Neither read nor write any cache for the requests made inside the block."""
    token = _bypass.set(True)
    try:
        yield
    finally:
        _bypass.reset(token)


def cache_bypassed() -> bool:
    """True inside :func:`cache_bypass`."""
    return _bypass.get()


# -- language detection ---------------------------------------------------------------

# Short, frequent words that are specific enough to tell these languages apart.
_STOPWORDS: dict[str, frozenset[str]] = {
    "fr": frozenset(
        "le la les des du un une est et vous nous je tu il elle ce cette quel quelle quels "
        "quelles pour avec dans sur pas que qui où comment combien pourquoi votre vos mon ma "
        "mes au aux est-ce êtes suis faire peut puis livrez".split()
    ),
    "en": frozenset(
        "the a an is are and you we i it this that what which who where how why your my "
        "do does did can could would should will to of in on for with not be have has".split()
    ),
    "es": frozenset(
        "el la los las un una es y usted ustedes yo tú qué cuál cuáles dónde cómo cuánto "
        "por para con en no su sus mi mis hay puedo puede está están".split()
    ),
    "pt": frozenset(
        "o a os as um uma é e você vocês eu que qual quais onde como quanto por para com "
        "em não seu sua seus meu minha há posso pode está estão vocês".split()
    ),
    "de": frozenset(
        "der die das ein eine ist und sie wir ich es was welche wer wo wie warum ihr mein "
        "zu von in auf für mit nicht kann können haben hat".split()
    ),
    "it": frozenset(
        "il lo la gli le un una è e voi noi io che quale quali dove come quanto perché "
        "per con in su non suo sua mio mia posso può sono".split()
    ),
}
_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?", re.UNICODE)
# Letters that only (or mostly) appear in one of these languages.
_LETTERS: dict[str, frozenset[str]] = {
    "fr": frozenset("éèêëàâçîïôûùœ"),
    "es": frozenset("ñ¿¡"),
    "pt": frozenset("ãõ"),
    "de": frozenset("äöüß"),
}


def detect_language(text: str) -> str:
    """Best-effort language of ``text`` (``"fr"``, ``"en"``...), ``"und"`` if unsure.

    A tiny, dependency-free detector: frequent words first, then letters specific to a
    language; plain unaccented Latin text with no other clue counts as English. Good
    enough to keep a French answer from being served to an English question. A wrong
    guess only causes a cache miss. Declare the language with :func:`language` when it
    matters more.
    """
    normalized = unicodedata.normalize("NFC", text).lower()
    words = _WORD.findall(normalized)
    if not words:
        return "und"
    scores = {code: sum(1 for w in words if w in vocab) for code, vocab in _STOPWORDS.items()}
    best = max(scores, key=lambda code: scores[code])
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] > 0 and ranked[0] > ranked[1]:
        return best
    letters = {code: sum(1 for ch in normalized if ch in chars) for code, chars in _LETTERS.items()}
    hinted = max(letters, key=lambda code: letters[code])
    if letters[hinted] > 0:
        return hinted
    if ranked[0] == 0 and all(word.isascii() for word in words):
        return "en"
    return "und"


# -- conversations ----------------------------------------------------------------------


def _messages(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Chat messages of a request, whatever the SDK (OpenAI, Anthropic, Mistral, Gemini)."""
    raw = request.get("messages")
    if raw is None:
        raw = request.get("contents")  # Gemini
    if not isinstance(raw, list):
        return []
    return [cast("Mapping[str, Any]", m) for m in cast("list[Any]", raw) if isinstance(m, Mapping)]


def depends_on_conversation(request: Mapping[str, Any]) -> bool:
    """True when the request carries a conversation: an earlier AI answer, a tool result,
    or more than one user message."""
    users = 0
    for message in _messages(request):
        role = message.get("role")
        if role in ("assistant", "model", "tool", "function"):
            return True
        if role == "user":
            users += 1
    return users > 1


def last_user_text(request: Mapping[str, Any]) -> str:
    """Text of the last user message (the question being asked)."""
    contents = request.get("contents")
    if isinstance(contents, str):  # Gemini shorthand: contents="Hello"
        return contents
    for message in reversed(_messages(request)):
        if message.get("role", "user") != "user":
            continue
        content = message.get("content", message.get("parts"))
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in cast("list[Any]", content):
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, Mapping):
                    text = cast("Mapping[str, Any]", part).get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return ""
    return ""


def _system_prompt(request: Mapping[str, Any]) -> Any:
    """Everything that plays the role of system instructions in a request."""
    found: list[Any] = []
    for name in ("system", "instructions"):  # Anthropic, OpenAI Responses
        if request.get(name) is not None:
            found.append(request[name])
    config = request.get("config")  # Gemini
    if isinstance(config, Mapping):
        instruction = cast("Mapping[str, Any]", config).get("system_instruction")
        if instruction is not None:
            found.append(instruction)
    for message in _messages(request):
        if message.get("role") in ("system", "developer"):
            found.append(message.get("content"))
    return found


def context_fingerprint(request: Mapping[str, Any], model: str | None) -> str:
    """Hash of (end customer, language, model, system instructions) for cache keys."""
    try:
        system = stable_hash(to_canonical(_system_prompt(request)))
    except UnhashableRequestError:
        system = "unhashable"
    declared = current_language()
    return stable_hash(
        {
            "customer": current_customer() or "",
            "language": declared or detect_language(last_user_text(request)),
            "model": model or "",
            "system": system,
        }
    )
