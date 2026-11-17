"""Router engine: picks the route of a request (first matching rule wins)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from caelum.router.config import Route, RouterConfig

__all__ = ["RouteDecision", "RouterEngine", "estimate_tokens", "request_texts"]


@dataclass(frozen=True)
class RouteDecision:
    """Where a request goes (``provider`` / ``model``) and which rule decided it."""

    route: str
    provider: str
    model: str


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    texts: list[str] = []
    if isinstance(content, list):
        for part in cast("list[Any]", content):
            if isinstance(part, Mapping):
                text = cast("Mapping[str, Any]", part).get("text")
                if isinstance(text, str):
                    texts.append(text)
    return "\n".join(texts)


def _messages(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = request.get("messages")
    items = cast("list[Any]", raw) if isinstance(raw, list) else []
    return [cast("Mapping[str, Any]", m) for m in items if isinstance(m, Mapping)]


def request_texts(request: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(system_text, user_text)`` of an OpenAI or Anthropic request."""
    system = [_content_text(request.get("system"))] if request.get("system") else []
    user: list[str] = []
    for message in _messages(request):
        text = _content_text(message.get("content"))
        if message.get("role") in ("system", "developer"):
            system.append(text)
        elif message.get("role") == "user":
            user.append(text)
    return "\n".join(system), "\n".join(user)


def estimate_tokens(request: Mapping[str, Any]) -> int:
    """Cheap token estimate of the whole prompt (about 4 characters per token).

    WHY not an exact tokenizer: routing thresholds are coarse, and exact counting would
    add latency (and, for some tokenizers, a network download) to every request.
    """
    characters = len(_content_text(request.get("system")))
    for message in _messages(request):
        characters += len(_content_text(message.get("content")))
    return (characters + 3) // 4


class RouterEngine:
    """Applies a :class:`RouterConfig` to requests."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    @staticmethod
    def _matches(route: Route, request: Mapping[str, Any], tokens: int) -> bool:
        match = route.match
        if match.messages_max_tokens is not None and tokens > match.messages_max_tokens:
            return False
        if match.messages_min_tokens is not None and tokens < match.messages_min_tokens:
            return False
        if match.models and request.get("model") not in match.models:
            return False
        if match.has_tools is not None and bool(request.get("tools")) != match.has_tools:
            return False
        if match.system_contains or match.user_contains:
            system, user = request_texts(request)
            if match.system_contains and not any(
                word.lower() in system.lower() for word in match.system_contains
            ):
                return False
            if match.user_contains and not any(
                word.lower() in user.lower() for word in match.user_contains
            ):
                return False
        return True

    def select_route(self, request: Mapping[str, Any]) -> RouteDecision | None:
        """Return the decision for ``request``, or None to keep the requested model."""
        tokens = estimate_tokens(request)
        for route in self.config.routes:
            if self._matches(route, request, tokens):
                return RouteDecision(route.name, route.provider, route.target)
        default = self.config.default
        if default is not None:
            return RouteDecision(default.name, default.provider, default.target)
        return None
