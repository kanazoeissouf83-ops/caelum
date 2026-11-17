"""Routing: each request goes to the cheapest capable model, per explicit rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import anthropic
import openai
import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion

import caelum
from caelum import Anthropic, OpenAI
from caelum.fallback import ProviderPool
from caelum.router import RouterConfig, RouterConfigError, RouterEngine, estimate_tokens
from tests.conftest import FakeServer

PROVIDER_ATTR = "_caelum_provider"
ROUTED_FROM_ATTR = "_caelum_routed_from"

CONFIG: dict[str, Any] = {
    "routes": [
        {
            "name": "cheap-reformulation",
            "match": {"messages_max_tokens": 500, "system_contains": ["reformulate", "REWRITE"]},
            "target": "claude-haiku-5-5",
            "provider": "anthropic",
        },
        {
            "name": "small-openai",
            "match": {"messages_max_tokens": 50, "models": ["gpt-4o"]},
            "target": "gpt-4o-mini",
            "provider": "openai",
        },
        {
            "name": "complex",
            "match": {"messages_min_tokens": 2000},
            "target": "gpt-4o",
            "provider": "openai",
        },
    ]
}


def raw_anthropic(server: FakeServer) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-test", http_client=server.http_client(), max_retries=0
    )


def openai_client(server: FakeServer, **kwargs: Any) -> OpenAI:
    options: dict[str, Any] = {
        "router_config": CONFIG,
        "provider_clients": {"anthropic": raw_anthropic(server)},
        **kwargs,
    }
    return OpenAI(api_key="sk-test", http_client=server.http_client(), **options)


def chat(c: OpenAI, content: str, system: str | None = None, **kwargs: Any) -> Any:
    messages: list[Any] = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    params: dict[str, Any] = {"model": "gpt-4o", "messages": messages, **kwargs}
    result: Any = cast(Any, c.chat.completions.create(**params))
    return result


# -- configuration ---------------------------------------------------------------


def test_config_loads_from_dict_json_and_yaml(tmp_path: Path) -> None:
    from_dict = RouterConfig.load(CONFIG)
    assert [route.name for route in from_dict.routes] == [
        "cheap-reformulation",
        "small-openai",
        "complex",
    ]
    assert from_dict.providers == {"anthropic", "openai"}
    json_path = tmp_path / "router.json"
    json_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    assert RouterConfig.load(json_path) == from_dict
    yaml_path = tmp_path / "router.yaml"
    yaml_path.write_text(
        "routes:\n"
        "  - name: cheap\n"
        "    match:\n"
        "      messages_max_tokens: 500\n"
        "    target: gpt-4o-mini\n"
        "    provider: openai\n"
        "default:\n"
        "  target: gpt-4o\n"
        "  provider: openai\n",
        encoding="utf-8",
    )
    loaded = RouterConfig.load(str(yaml_path))
    assert loaded.default is not None and loaded.default.target == "gpt-4o"
    assert RouterConfig.load(loaded) is loaded


def test_yaml_without_pyyaml_gives_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "r.yml"
    path.write_text("routes: []\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match=r"caelum-ai\[yaml\]"):
        RouterConfig.load(path)


INVALID_CONFIGS: list[Any] = [
    {"routes": [{"name": "a", "target": "m", "provider": "google"}]},
    {"routes": [{"name": "a", "target": "m", "provider": "openai", "extra": 1}]},
    {"routes": [{"name": "a", "target": "m", "provider": "openai", "match": {"oops": 1}}]},
    {"routes": [{"name": "a", "provider": "openai"}]},
    {"routes": [{"target": "m", "provider": "openai"}]},
    {
        "routes": [
            {"name": "a", "target": "m", "provider": "openai"},
            {"name": "a", "target": "n", "provider": "openai"},
        ]
    },
    {
        "routes": [
            {
                "name": "a",
                "target": "m",
                "provider": "openai",
                "match": {"messages_min_tokens": 10, "messages_max_tokens": 5},
            }
        ]
    },
    {"routes": [{"name": "a", "target": "m", "provider": "openai", "match": {"has_tools": 1}}]},
    {
        "routes": [
            {"name": "a", "target": "m", "provider": "openai", "match": {"messages_max_tokens": -1}}
        ]
    },
    {"routes": [{"name": "a", "target": "m", "provider": "openai", "match": {"models": [1]}}]},
    {"routes": "nope"},
    {"routes": []},
    {"unknown": []},
    ["not", "a", "mapping"],
]


@pytest.mark.parametrize("config", INVALID_CONFIGS)
def test_invalid_configs_are_rejected(config: Any) -> None:
    with pytest.raises(RouterConfigError):
        RouterConfig.from_dict(config)


def test_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "router.toml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RouterConfigError):
        RouterConfig.load(path)


# -- rule matching ---------------------------------------------------------------


def test_first_matching_route_wins_and_default_applies() -> None:
    engine = RouterEngine(
        RouterConfig.from_dict(
            {**CONFIG, "default": {"target": "gpt-4o-mini", "provider": "openai"}}
        )
    )
    short_rewrite: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Please Rewrite the text."},
            {"role": "user", "content": "hi"},
        ],
    }
    decision = engine.select_route(short_rewrite)
    assert decision is not None and decision.route == "cheap-reformulation"
    long_request: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x" * 9000}],
    }
    decision = engine.select_route(long_request)
    assert decision is not None and decision.route == "complex"
    medium: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x" * 1000}],
    }
    decision = engine.select_route(medium)
    assert decision is not None and decision.route == "default"


def test_conditions() -> None:
    config = RouterConfig.from_dict(
        {
            "routes": [
                {
                    "name": "tools",
                    "match": {"has_tools": True, "user_contains": "refund"},
                    "target": "gpt-4o",
                    "provider": "openai",
                }
            ]
        }
    )
    engine = RouterEngine(config)
    base: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "A REFUND please"}],
    }
    assert engine.select_route(base) is None  # no tools
    assert engine.select_route({**base, "tools": [{"type": "function"}]}) is not None
    other: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [1],
    }
    assert engine.select_route(other) is None


def test_estimate_tokens_counts_all_text() -> None:
    request: dict[str, Any] = {
        "system": "abcd",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "abcdefgh"}]},
            {"role": "assistant", "content": "abcd"},
            "ignored",
        ],
    }
    assert estimate_tokens(request) == 4


# -- integration --------------------------------------------------------------------


def test_same_provider_route_overrides_the_model(server: FakeServer) -> None:
    c = openai_client(server)
    response = chat(c, "hi")
    assert server.requests[-1]["body"]["model"] == "gpt-4o-mini"
    assert caelum.routed_to(response) == "gpt-4o-mini"
    routed_from: Any = getattr(response, ROUTED_FROM_ATTR)
    assert routed_from == "gpt-4o"
    assert c.metrics.summary()["routed"] == 1
    assert c.metrics.summary()["requests_by_model"] == {"gpt-4o": 1}


def test_cross_provider_route_returns_an_openai_object(server: FakeServer) -> None:
    c = openai_client(server)
    response = chat(c, "Some text", system="Reformulate this.", temperature=0.4)
    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "answer #1"
    assert response.model == "claude-haiku-5-5"
    sent = server.requests[-1]
    assert sent["path"].endswith("/messages")
    assert sent["body"]["system"] == "Reformulate this."
    assert "temperature" not in sent["body"]  # rejected by current Claude models
    provider: Any = getattr(response, PROVIDER_ATTR)
    assert provider == "anthropic"
    assert c.metrics.summary()["cost_usd"] == pytest.approx((1000 * 0.10 + 500 * 0.50) / 1e6)


def test_untranslatable_request_keeps_the_original_provider(server: FakeServer) -> None:
    c = openai_client(server)
    chat(c, "Some text", system="Reformulate this.", n=2)
    assert server.requests[-1]["path"].endswith("/chat/completions")
    assert server.requests[-1]["body"]["model"] == "gpt-4o"


def test_streams_are_never_routed_across_providers(server: FakeServer) -> None:
    c = openai_client(server)
    list(chat(c, "Some text", system="Reformulate this.", stream=True))
    assert server.requests[-1]["path"].endswith("/chat/completions")


def test_routed_answers_are_cached_under_their_destination(
    server: FakeServer, cache_path: Path
) -> None:
    c = openai_client(server, cache=True, cache_path=cache_path)
    chat(c, "Some text", system="Reformulate this.")
    second = chat(c, "Some text", system="Reformulate this.")
    assert caelum.cache_hit(second)
    assert isinstance(second, ChatCompletion)
    plain = OpenAI(
        api_key="sk-test", http_client=server.http_client(), cache=True, cache_path=cache_path
    )
    assert not caelum.cache_hit(chat(plain, "Some text", system="Reformulate this."))


def test_anthropic_client_can_route_to_openai(server: FakeServer) -> None:
    raw_openai = openai.OpenAI(api_key="sk-test", http_client=server.http_client(), max_retries=0)
    c = Anthropic(
        api_key="sk-ant-test",
        http_client=server.http_client(),
        router_config={"default": {"target": "gpt-4o-mini", "provider": "openai"}},
        provider_clients={"openai": raw_openai},
    )
    message = c.messages.create(
        model="claude-opus-5-5", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    assert isinstance(message, Message)
    assert message.content[0].type == "text"
    assert server.requests[-1]["path"].endswith("/chat/completions")
    assert server.requests[-1]["body"]["max_completion_tokens"] == 100


def test_unknown_provider_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        ProviderPool(native="openai", is_async=False, keys={"google": "x"})


def test_missing_provider_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"caelum-ai\[anthropic\]"):
        OpenAI(api_key="sk-test", router_config=CONFIG)


def test_provider_pool_creates_clients_lazily_with_keys() -> None:
    pool = ProviderPool(native="openai", is_async=False, keys={"anthropic": "sk-ant-from-keys"})
    client = pool.get("anthropic")
    assert isinstance(client, anthropic.Anthropic)
    assert client.api_key == "sk-ant-from-keys"
    assert pool.get("anthropic") is client
    async_pool = ProviderPool(native="anthropic", is_async=True, keys={"openai": "sk-x"})
    assert isinstance(async_pool.get("openai"), openai.AsyncOpenAI)
