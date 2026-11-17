"""Cross-provider translation between the OpenAI Chat Completions and Anthropic Messages
formats (requests and responses).

Used by routing (send a request to a cheaper model of another provider) and fallback
(switch provider when one is down). Supported: text, images (URL and base64), system
prompts, function tools, tool calls and tool results, stop sequences, max tokens.

Anything that cannot be translated faithfully raises :class:`UntranslatableRequestError`
(the caller then keeps the original provider): ``n > 1``, logprobs, audio, structured
``response_format``, forced tool choice towards Claude models (rejected by the Claude API),
server tools, streaming.

Dropped on purpose when sending to Anthropic: ``temperature``, ``top_p`` (rejected by the
Claude API since 2026-10-21), ``seed``, presence/frequency penalties, ``reasoning_effort``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, cast

from caelum.providers import dialect

__all__ = [
    "UntranslatableRequestError",
    "anthropic_to_openai_request",
    "anthropic_to_openai_response",
    "openai_to_anthropic_request",
    "openai_to_anthropic_response",
    "translate_request",
    "translate_response",
]

DEFAULT_ANTHROPIC_MAX_TOKENS = 4096


class UntranslatableRequestError(ValueError):
    """The request uses a feature that has no faithful equivalent on the target."""


# Parameters with no effect on the answer, or deliberately dropped (see module doc).
_OPENAI_DROPPED = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "verbosity",
        "store",
        "metadata",
        "stream_options",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "parallel_tool_calls",
        "user",
        "stream",
    }
)
_OPENAI_UNTRANSLATABLE = frozenset(
    {
        "n",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "audio",
        "modalities",
        "prediction",
        "functions",
        "function_call",
        "web_search_options",
        "moderation",
    }
)
_ANTHROPIC_DROPPED = frozenset(
    {"thinking", "service_tier", "cache_control", "diagnostics", "inference_geo", "speed"}
)
_TRANSPORT = frozenset({"timeout", "extra_headers", "extra_query", "extra_body"})


def _as_list(value: Any) -> list[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else []


def _as_map(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UntranslatableRequestError(f"unexpected value: {value!r:.80}")
    return cast("Mapping[str, Any]", value)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = [_as_map(part) for part in _as_list(content)]
    if any(part.get("type") != "text" for part in parts):
        raise UntranslatableRequestError("non-text content where only text is supported")
    return "\n".join(str(part.get("text", "")) for part in parts)


def _is_neutral(name: str, value: Any) -> bool:
    """True if an otherwise untranslatable parameter is set to its no-op value."""
    # WHY: compare with `is`, because True == 1 in Python (logprobs=True is not n=1).
    if name == "n":
        return value is None or (type(value) is int and value == 1)
    return value is None or value is False


def _request_arguments(arguments: Any) -> Any:
    """Tool-call arguments of a request message (JSON text in OpenAI, object in Anthropic)."""
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError) as exc:
        # WHY: e.g. a tool call truncated by max_tokens and echoed back in the history.
        raise UntranslatableRequestError("tool call arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        # WHY: Anthropic requires tool_use.input to be an object.
        raise UntranslatableRequestError("tool call arguments are not a JSON object")
    return cast("dict[str, Any]", parsed)


def _response_arguments(arguments: Any) -> Any:
    """Tool-call arguments of a response; invalid JSON is kept as raw text, never lost."""
    try:
        return json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {"_raw_arguments": str(arguments)}


def _common(request: Mapping[str, Any], out: dict[str, Any]) -> None:
    for name in ("extra_body", "extra_query"):
        if request.get(name):
            # WHY: provider-specific fields would be sent to another API and rejected.
            raise UntranslatableRequestError(f"{name} is specific to the original provider")
    for name in _TRANSPORT:
        if name in request:
            out[name] = request[name]


# ---------------------------------------------------------------------------
# OpenAI request -> Anthropic request
# ---------------------------------------------------------------------------


def _openai_image_to_anthropic(part: Mapping[str, Any]) -> dict[str, Any]:
    url = str(_as_map(part.get("image_url")).get("url", ""))
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _openai_user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks: list[dict[str, Any]] = []
    for raw in _as_list(content):
        part = _as_map(raw)
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_url":
            blocks.append(_openai_image_to_anthropic(part))
        else:
            raise UntranslatableRequestError(f"unsupported content part: {part.get('type')}")
    return blocks


def openai_to_anthropic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Translate ``chat.completions.create`` kwargs into ``messages.create`` kwargs."""
    if request.get("stream") is True:
        raise UntranslatableRequestError("cross-provider streaming is not supported yet")
    for name in request:
        if name in _OPENAI_UNTRANSLATABLE and not _is_neutral(name, request[name]):
            raise UntranslatableRequestError(f"parameter {name!r} has no Anthropic equivalent")
    response_format = request.get("response_format")
    if response_format is not None and _as_map(response_format).get("type") != "text":
        raise UntranslatableRequestError("structured response_format is not translated")

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if messages and messages[-1]["role"] == role:  # Anthropic requires alternation
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for raw in _as_list(request.get("messages")):
        message = _as_map(raw)
        role = message.get("role")
        if role in ("system", "developer"):
            system_parts.append(_text_of(message.get("content")))
        elif role == "user":
            push("user", _openai_user_blocks(message.get("content")))
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _text_of(message.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for raw_call in _as_list(message.get("tool_calls")):
                tool_call = _as_map(raw_call)
                function = _as_map(tool_call.get("function"))
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "input": _request_arguments(function.get("arguments")),
                    }
                )
            if not blocks:
                # WHY: Anthropic rejects an assistant turn with no content blocks.
                raise UntranslatableRequestError("empty assistant message")
            push("assistant", blocks)
        elif role == "tool":
            push(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id"),
                        "content": _text_of(message.get("content")),
                    }
                ],
            )
        else:
            raise UntranslatableRequestError(f"unsupported message role: {role!r}")

    out: dict[str, Any] = {
        "model": request.get("model"),
        "messages": messages,
        "max_tokens": request.get("max_completion_tokens")
        or request.get("max_tokens")
        or DEFAULT_ANTHROPIC_MAX_TOKENS,
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    stop = request.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(_as_list(stop))
    user = request.get("user")
    if isinstance(user, str):
        out["metadata"] = {"user_id": user}
    tools = _as_list(request.get("tools"))
    if tools:
        converted: list[dict[str, Any]] = []
        for raw_tool in tools:
            tool = _as_map(raw_tool)
            if tool.get("type") != "function":
                raise UntranslatableRequestError("only function tools are translated")
            function = _as_map(tool.get("function"))
            converted.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters") or {"type": "object"},
                }
            )
        out["tools"] = converted
    tool_choice = request.get("tool_choice")
    if tool_choice in ("auto", "none"):
        out["tool_choice"] = {"type": tool_choice}
    elif tool_choice is not None:
        # WHY: current Claude models reject forced tool use (any / tool) with a 400.
        raise UntranslatableRequestError("forced tool_choice is not supported by Claude models")
    for name in request:
        known = (
            {"model", "messages", "max_tokens", "max_completion_tokens", "stop", "tools"}
            | {"tool_choice", "response_format"}
            | _OPENAI_DROPPED
            | _OPENAI_UNTRANSLATABLE
            | _TRANSPORT
        )
        if name not in known:
            raise UntranslatableRequestError(f"unknown parameter {name!r}")
    _apply_reasoning(request, out)
    _common(request, out)
    return out


# WHY: an OpenAI Chat Completions request without ``reasoning_effort`` expects a direct
# answer within ``max_tokens``. Current Claude models think by default, and with a small
# budget the thinking can consume every token, leaving an EMPTY answer (observed in the
# real-API tests). When translating, ask Claude for a direct answer, in the way each model
# accepts (per Anthropic's model documentation, 2026-10):
_THINKING_OFF: dict[str, dict[str, str]] = {
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},  # "disabled" is rejected on this model
    "claude-sonnet-5": {"type": "disabled"},
    "claude-opus-5": {"type": "disabled"},
}
# Models whose thinking cannot be turned off: use the lowest effort instead.
_LOWEST_EFFORT_ONLY = frozenset({"claude-opus-5-5", "claude-fable-5-1", "claude-fable-5"})
_EFFORT_FROM_OPENAI = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


def _apply_reasoning(request: Mapping[str, Any], out: dict[str, Any]) -> None:
    """Map OpenAI reasoning intent onto Claude thinking / effort settings."""
    model = str(request.get("model", ""))
    requested = request.get("reasoning_effort")
    if isinstance(requested, str) and requested in _EFFORT_FROM_OPENAI:
        # The caller asked for reasoning: keep Claude's thinking, at the matching effort.
        out["output_config"] = {"effort": _EFFORT_FROM_OPENAI[requested]}
        return
    if model in _THINKING_OFF:
        out["thinking"] = dict(_THINKING_OFF[model])
    elif model in _LOWEST_EFFORT_ONLY:
        out["output_config"] = {"effort": "low"}


# ---------------------------------------------------------------------------
# Anthropic request -> OpenAI request
# ---------------------------------------------------------------------------


def _anthropic_image_to_openai(block: Mapping[str, Any]) -> dict[str, Any]:
    source = _as_map(block.get("source"))
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type')};base64,{source.get('data')}"
    elif source.get("type") == "url":
        url = str(source.get("url"))
    else:
        raise UntranslatableRequestError("unsupported image source")
    return {"type": "image_url", "image_url": {"url": url}}


def anthropic_to_openai_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Translate ``messages.create`` kwargs into ``chat.completions.create`` kwargs."""
    if request.get("stream") is True:
        raise UntranslatableRequestError("cross-provider streaming is not supported yet")
    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    for raw in _as_list(request.get("messages")):
        message = _as_map(raw)
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if role == "user":
            parts: list[dict[str, Any]] = []
            for raw_block in _as_list(content):
                block = _as_map(raw_block)
                kind = block.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif kind == "image":
                    parts.append(_anthropic_image_to_openai(block))
                elif kind == "tool_result":
                    result = _text_of(block.get("content"))
                    if block.get("is_error") is True:
                        # WHY: OpenAI has no error flag; keep the information in the text.
                        result = f"[tool error] {result}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": result,
                        }
                    )
                else:
                    raise UntranslatableRequestError(f"unsupported content block: {kind}")
            if parts:
                messages.append({"role": "user", "content": parts})
        elif role == "assistant":
            texts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for raw_block in _as_list(content):
                block = _as_map(raw_block)
                kind = block.get("type")
                if kind == "text":
                    texts.append(str(block.get("text", "")))
                elif kind == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif kind in ("thinking", "redacted_thinking"):
                    continue  # model-specific reasoning cannot be replayed elsewhere
                else:
                    raise UntranslatableRequestError(f"unsupported content block: {kind}")
            converted: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if tool_calls:
                converted["tool_calls"] = tool_calls
            messages.append(converted)
        else:
            raise UntranslatableRequestError(f"unsupported message role: {role!r}")

    out: dict[str, Any] = {"model": request.get("model"), "messages": messages}
    if request.get("max_tokens"):
        out["max_completion_tokens"] = request["max_tokens"]
    if request.get("stop_sequences"):
        out["stop"] = list(_as_list(request["stop_sequences"]))
    for name in ("temperature", "top_p"):
        if name in request:
            out[name] = request[name]
    metadata = request.get("metadata")
    if isinstance(metadata, Mapping) and "user_id" in metadata:
        out["user"] = cast("Mapping[str, Any]", metadata)["user_id"]
    tools = _as_list(request.get("tools"))
    if tools:
        converted_tools: list[dict[str, Any]] = []
        for raw_tool in tools:
            tool = _as_map(raw_tool)
            if "input_schema" not in tool:
                raise UntranslatableRequestError("server tools are not translated")
            converted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema"),
                    },
                }
            )
        out["tools"] = converted_tools
    tool_choice = request.get("tool_choice")
    if tool_choice is not None:
        choice = _as_map(tool_choice)
        kind = choice.get("type")
        mapping = {"auto": "auto", "none": "none", "any": "required"}
        if kind in mapping:
            out["tool_choice"] = mapping[str(kind)]
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        else:
            raise UntranslatableRequestError(f"unsupported tool_choice {kind!r}")
    known = (
        {"model", "messages", "system", "max_tokens", "stop_sequences", "temperature", "top_p"}
        | {"metadata", "tools", "tool_choice", "stream"}
        | _ANTHROPIC_DROPPED
        | _TRANSPORT
    )
    for name in request:
        if name not in known:
            raise UntranslatableRequestError(f"parameter {name!r} has no OpenAI equivalent")
    _common(request, out)
    return out


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

_ANTHROPIC_TO_OPENAI_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "stop",
}
_OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def anthropic_to_openai_response(message: Any) -> Any:
    """Turn an Anthropic ``Message`` into an OpenAI ``ChatCompletion``."""
    from openai.types.chat import ChatCompletion

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
    assistant: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    usage = message.usage
    return ChatCompletion.model_validate(
        {
            "id": message.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": message.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": _ANTHROPIC_TO_OPENAI_FINISH.get(
                        str(message.stop_reason), "stop"
                    ),
                    "message": assistant,
                }
            ],
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            },
        }
    )


def openai_to_anthropic_response(completion: Any) -> Any:
    """Turn an OpenAI ``ChatCompletion`` into an Anthropic ``Message``."""
    from anthropic.types import Message

    choice = completion.choices[0]
    blocks: list[dict[str, Any]] = []
    stop_reason = _OPENAI_TO_ANTHROPIC_STOP.get(str(choice.finish_reason), "end_turn")
    if choice.message.content:
        blocks.append({"type": "text", "text": choice.message.content})
    refusal = getattr(choice.message, "refusal", None)
    if isinstance(refusal, str) and refusal and not choice.message.content:
        blocks.append({"type": "text", "text": refusal})
        stop_reason = "refusal"
    for tool_call in cast("list[Any]", choice.message.tool_calls or []):
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": function.name,
                "input": _response_arguments(function.arguments),
            }
        )
    usage = completion.usage
    return Message.model_validate(
        {
            "id": completion.id,
            "type": "message",
            "role": "assistant",
            "model": completion.model,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        }
    )


def translate_request(source: str, target: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Translate request kwargs from ``source`` provider format to ``target``."""
    source, target = dialect(source), dialect(target)
    if source == target:
        return dict(request)
    if (source, target) == ("openai", "anthropic"):
        return openai_to_anthropic_request(request)
    if (source, target) == ("anthropic", "openai"):
        return anthropic_to_openai_request(request)
    raise UntranslatableRequestError(f"no translation from {source} to {target}")


def translate_response(source: str, target: str, response: Any) -> Any:
    """Translate a response object from ``source`` provider format to ``target``."""
    source, target = dialect(source), dialect(target)
    if source == target:
        return response
    if (source, target) == ("anthropic", "openai"):
        return anthropic_to_openai_response(response)
    if (source, target) == ("openai", "anthropic"):
        return openai_to_anthropic_response(response)
    raise UntranslatableRequestError(f"no translation from {source} to {target}")
