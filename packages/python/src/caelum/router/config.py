"""Router configuration: ordered rules sending each request to the cheapest capable model.

Example (YAML, or the same structure as a Python dict / JSON file)::

    routes:
      - name: cheap-reformulation
        match:
          messages_max_tokens: 500
          system_contains: ["reformulate", "rewrite", "paraphrase"]
        target: claude-haiku-5-5
        provider: anthropic
      - name: complex-reasoning
        match:
          messages_min_tokens: 2000
        target: gpt-4o
        provider: openai
    default:            # optional: without it, unmatched requests keep their model
      target: gpt-4o-mini
      provider: openai

Conditions inside one ``match`` must ALL hold. Text conditions (``*_contains``) match if
ANY of the listed words appears (case-insensitive). Routes are tried in order: the first
match wins.

YAML files require PyYAML (``pip install "caelum-ai[yaml]"``); JSON files and dicts need
nothing extra. Validation is strict: unknown keys or wrong types raise immediately.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from caelum.providers import SUPPORTED_PROVIDERS
from caelum.utils import missing_extra_message

__all__ = ["SUPPORTED_PROVIDERS", "Route", "RouteMatch", "RouterConfig", "RouterConfigError"]


_MATCH_KEYS = frozenset(
    {
        "messages_max_tokens",
        "messages_min_tokens",
        "system_contains",
        "user_contains",
        "models",
        "has_tools",
    }
)


class RouterConfigError(ValueError):
    """Invalid router configuration."""


def _strings(value: Any, where: str) -> tuple[str, ...]:
    """A string or a non-empty list of non-empty strings (absent = no condition)."""
    if value is None:
        return ()
    items: list[Any] = [value] if isinstance(value, str) else cast("list[Any]", value)
    if (
        not isinstance(value, (str, list))
        or not items
        or not all(isinstance(v, str) and v for v in items)
    ):
        # WHY: "" or [] would silently match every request (or none).
        raise RouterConfigError(f"{where} must be a non-empty string or list of strings")
    return tuple(cast("list[str]", items))


def _optional_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RouterConfigError(f"{where} must be a non-negative integer")
    return value


def _provider(value: Any, where: str) -> str:
    if value not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise RouterConfigError(f"{where}: unsupported provider {value!r} (supported: {supported})")
    return cast(str, value)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouterConfigError(f"{where} must be a mapping")
    return cast("Mapping[str, Any]", value)


def _check_keys(data: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise RouterConfigError(f"{where}: unknown key(s) {sorted(unknown)}")


@dataclass(frozen=True)
class RouteMatch:
    """Conditions that must all hold for a route to apply."""

    messages_max_tokens: int | None = None
    messages_min_tokens: int | None = None
    system_contains: tuple[str, ...] = ()
    user_contains: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    has_tools: bool | None = None

    @classmethod
    def parse(cls, data: Any, where: str) -> RouteMatch:
        """Validate and build the ``match`` section of a route."""
        mapping = _mapping(data, where)
        _check_keys(mapping, _MATCH_KEYS, where)
        if not mapping:
            raise RouterConfigError(
                f"{where} has no condition: it would match every request (use `default`)"
            )
        has_tools = mapping.get("has_tools")
        if has_tools is not None and not isinstance(has_tools, bool):
            raise RouterConfigError(f"{where}.has_tools must be true or false")

        def key(name: str) -> str:
            return f"{where}.{name}"

        match = cls(
            messages_max_tokens=_optional_int(
                mapping.get("messages_max_tokens"), key("messages_max_tokens")
            ),
            messages_min_tokens=_optional_int(
                mapping.get("messages_min_tokens"), key("messages_min_tokens")
            ),
            system_contains=_strings(mapping.get("system_contains"), key("system_contains")),
            user_contains=_strings(mapping.get("user_contains"), key("user_contains")),
            models=_strings(mapping.get("models"), key("models")),
            has_tools=has_tools,
        )
        low, high = match.messages_min_tokens, match.messages_max_tokens
        if low is not None and high is not None and low > high:
            raise RouterConfigError(f"{where}: messages_min_tokens > messages_max_tokens")
        return match


@dataclass(frozen=True)
class Route:
    """A named rule: if ``match`` holds, send the request to ``provider`` / ``target``."""

    name: str
    target: str
    provider: str
    match: RouteMatch = field(default_factory=RouteMatch)

    @classmethod
    def parse(cls, data: Any, where: str, *, default: bool = False) -> Route:
        """Validate and build one route (or the ``default`` section)."""
        mapping = _mapping(data, where)
        allowed: set[str] = {"name", "target", "provider"}
        if not default:
            allowed.add("match")
        _check_keys(mapping, frozenset(allowed), where)
        name = mapping.get("name", "default" if default else None)
        if not isinstance(name, str) or not name:
            raise RouterConfigError(f"{where}.name must be a non-empty string")
        target = mapping.get("target")
        if not isinstance(target, str) or not target:
            raise RouterConfigError(f"{where}.target must be a model name")
        match = (
            RouteMatch() if default else RouteMatch.parse(mapping.get("match"), f"{where}.match")
        )
        provider = _provider(mapping.get("provider"), where)
        return cls(name=name, target=target, provider=provider, match=match)


@dataclass(frozen=True)
class RouterConfig:
    """Validated router configuration."""

    routes: tuple[Route, ...]
    default: Route | None = None

    @classmethod
    def from_dict(cls, data: Any) -> RouterConfig:
        """Build a configuration from a dict (same structure as the YAML file)."""
        mapping = _mapping(data, "router_config")
        _check_keys(mapping, frozenset({"routes", "default"}), "router_config")
        raw_routes = mapping.get("routes", [])
        if not isinstance(raw_routes, list):
            raise RouterConfigError("router_config.routes must be a list")
        routes = tuple(
            Route.parse(item, f"routes[{index}]")
            for index, item in enumerate(cast("list[Any]", raw_routes))
        )
        names = [route.name for route in routes]
        if len(names) != len(set(names)):
            raise RouterConfigError("route names must be unique")
        default_data = mapping.get("default")
        default = None
        if default_data is not None:
            default = Route.parse(default_data, "default", default=True)
        if not routes and default is None:
            raise RouterConfigError("router_config needs at least one route or a default")
        return cls(routes=routes, default=default)

    @classmethod
    def load(
        cls, source: RouterConfig | Mapping[str, Any] | str | os.PathLike[str]
    ) -> RouterConfig:
        """Load a configuration from a RouterConfig, a dict, or a ``.yaml``/``.json`` file."""
        if isinstance(source, RouterConfig):
            return source
        if isinstance(source, Mapping):
            return cls.from_dict(source)
        if not isinstance(source, (str, os.PathLike)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise RouterConfigError(
                "router_config must be a dict, a RouterConfig, or the path of a .yaml/.json file"
            )
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls.from_dict(json.loads(text))
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                message = missing_extra_message("yaml", "caelum router YAML files")
                raise ImportError(message) from exc
            return cls.from_dict(yaml.safe_load(text))
        raise RouterConfigError(f"unsupported router config file type: {suffix!r}")

    @property
    def providers(self) -> frozenset[str]:
        """Every provider referenced by the configuration."""
        every = [*self.routes, *([self.default] if self.default else [])]
        return frozenset(route.provider for route in every)
