"""Cross-provider translation must be faithful, or refuse."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion

from caelum.translate import (
    UntranslatableRequestError,
    anthropic_to_openai_request,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
    openai_to_anthropic_response,
    translate_request,
    translate_response,
)

TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_order",
        "description": "Look up an order",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
    },
}


def test_openai_to_anthropic_full_conversation() -> None:
    request: dict[str, Any] = {
        "model": "claude-haiku-5-5",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": "Use English."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Where is order 42?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.jpg"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"id": "42"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "shipped"},
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "seed": 1,
        "stop": "END",
        "max_completion_tokens": 300,
        "max_tokens": 999,
        "tools": [TOOL],
        "tool_choice": "auto",
        "user": "u-1",
        "timeout": 12,
    }
    out = openai_to_anthropic_request(request)
    assert out["system"] == "Be brief.\n\nUse English."
    assert out["max_tokens"] == 300
    assert out["stop_sequences"] == ["END"]
    assert out["metadata"] == {"user_id": "u-1"}
    assert out["timeout"] == 12
    assert "temperature" not in out and "top_p" not in out and "seed" not in out
    assert out["tool_choice"] == {"type": "auto"}
    assert out["tools"] == [
        {
            "name": "get_order",
            "description": "Look up an order",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
        }
    ]
    user, assistant, tool_result = out["messages"]
    assert user["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert user["content"][2]["source"] == {"type": "url", "url": "https://x.test/a.jpg"}
    assert assistant["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "get_order", "input": {"id": "42"}}
    ]
    assert tool_result == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "shipped"}],
    }


def test_consecutive_same_role_messages_are_merged() -> None:
    out = openai_to_anthropic_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        }
    )
    assert out["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    ]
    assert out["max_tokens"] == 4096


UNTRANSLATABLE_OPENAI: list[dict[str, Any]] = [
    {"n": 2},
    {"logprobs": True},
    {"stream": True},
    {"tool_choice": "required"},
    {"tool_choice": {"type": "function", "function": {"name": "get_order"}}},
    {"response_format": {"type": "json_object"}},
    {"something_new": 1},
    {"tools": [{"type": "web_search"}]},
    {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]},
    {"messages": [{"role": "function", "content": "x"}]},
]


@pytest.mark.parametrize("extra", UNTRANSLATABLE_OPENAI)
def test_openai_features_without_equivalent_are_refused(extra: dict[str, Any]) -> None:
    request: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(UntranslatableRequestError):
        openai_to_anthropic_request({**request, **extra})


def test_openai_neutral_values_are_accepted() -> None:
    request: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "n": 1,
        "logprobs": False,
        "response_format": {"type": "text"},
    }
    assert openai_to_anthropic_request(request)["messages"][0]["role"] == "user"


def test_anthropic_to_openai_full_conversation() -> None:
    request: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "max_tokens": 200,
        "system": [{"type": "text", "text": "Be brief."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "x"},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "tu_1", "name": "get_order", "input": {"id": "1"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}],
            },
            {"role": "user", "content": "thanks"},
        ],
        "stop_sequences": ["END"],
        "metadata": {"user_id": "u-9"},
        "tools": [{"name": "get_order", "description": "d", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
        "thinking": {"type": "adaptive"},
    }
    out = anthropic_to_openai_request(request)
    assert out["messages"][0] == {"role": "system", "content": "Be brief."}
    assert out["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }
    assistant = out["messages"][2]
    assert assistant["content"] == "Checking."
    assert assistant["tool_calls"][0]["function"] == {
        "name": "get_order",
        "arguments": '{"id": "1"}',
    }
    assert out["messages"][3] == {"role": "tool", "tool_call_id": "tu_1", "content": "ok"}
    assert out["messages"][4] == {"role": "user", "content": "thanks"}
    assert out["max_completion_tokens"] == 200
    assert out["stop"] == ["END"]
    assert out["user"] == "u-9"
    assert out["tool_choice"] == "required"
    assert out["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert "thinking" not in out


def test_anthropic_specific_tool_choice() -> None:
    out = anthropic_to_openai_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "tool", "name": "f"},
        }
    )
    assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}


UNTRANSLATABLE_ANTHROPIC: list[dict[str, Any]] = [
    {"tools": [{"type": "web_search_20260209", "name": "web_search"}]},
    {"output_config": {"format": {"type": "json_schema"}}},
    {"stream": True},
    {"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "file"}}]}]},
]


@pytest.mark.parametrize("extra", UNTRANSLATABLE_ANTHROPIC)
def test_anthropic_features_without_equivalent_are_refused(extra: dict[str, Any]) -> None:
    request: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(UntranslatableRequestError):
        anthropic_to_openai_request({**request, **extra})


def make_message(**overrides: Any) -> Message:
    data: dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "text", "text": "Order 42 "},
            {"type": "text", "text": "has shipped."},
            {"type": "tool_use", "id": "tu_1", "name": "track", "input": {"id": "42"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        **overrides,
    }
    return Message.model_validate(data)


def test_anthropic_message_to_chat_completion() -> None:
    completion = anthropic_to_openai_response(make_message())
    assert isinstance(completion, ChatCompletion)
    choice = completion.choices[0]
    assert choice.message.content == "Order 42 has shipped."
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert completion.model == "claude-haiku-5-5"
    assert completion.usage is not None and completion.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [("end_turn", "stop"), ("max_tokens", "length"), ("refusal", "content_filter")],
)
def test_stop_reason_mapping(stop_reason: str, finish: str) -> None:
    message = make_message(stop_reason=stop_reason, content=[{"type": "text", "text": "x"}])
    assert anthropic_to_openai_response(message).choices[0].finish_reason == finish


def test_chat_completion_to_anthropic_message() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "partial",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"a": 1}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
    )
    message = openai_to_anthropic_response(completion)
    assert isinstance(message, Message)
    assert message.stop_reason == "max_tokens"
    assert message.content[0].type == "text"
    assert message.content[1].type == "tool_use"
    assert message.usage.input_tokens == 7


def test_dispatch_helpers() -> None:
    request: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    assert translate_request("openai", "openai", request) == request
    assert translate_request("anthropic", "openai", request)["model"] == "m"
    sentinel = object()
    assert translate_response("openai", "openai", sentinel) is sentinel
    with pytest.raises(UntranslatableRequestError):
        translate_request("openai", "google", request)
    with pytest.raises(UntranslatableRequestError):
        translate_response("google", "openai", sentinel)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-5-5", {"thinking": {"type": "disabled"}}),
        ("claude-sonnet-5-5", {"thinking": {"type": "between_tools"}}),
        ("claude-opus-5-5", {"output_config": {"effort": "low"}}),
        ("claude-some-future-model", {}),
    ],
)
def test_direct_answer_is_requested_from_claude(model: str, expected: dict[str, Any]) -> None:
    """Without reasoning_effort, Claude must not spend the whole budget thinking."""
    out = openai_to_anthropic_request(
        {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert {k: out[k] for k in ("thinking", "output_config") if k in out} == expected


def test_openai_reasoning_effort_keeps_claude_thinking() -> None:
    out = openai_to_anthropic_request(
        {
            "model": "claude-haiku-5-5",
            "reasoning_effort": "minimal",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert out["output_config"] == {"effort": "low"}
    assert "thinking" not in out
