"""Shared test fixtures.

Tests never call real APIs. Instead, the *real* provider SDKs are plugged into an
in-memory fake server through ``httpx2.MockTransport``: requests go through the full SDK
stack (serialization, parsing, streaming, error mapping) without touching the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest


@dataclass
class FakeServer:
    """In-memory fake of the OpenAI and Anthropic HTTP APIs.

    Every chat/messages answer contains a call counter (``"answer #N"``) so tests can
    tell a fresh API answer from a cached one.
    """

    requests: list[dict[str, Any]] = field(default_factory=lambda: [])
    # Queue of HTTP status codes to return before succeeding (for retry tests).
    failures: list[int] = field(default_factory=lambda: [])
    failure_headers: dict[str, str] = field(default_factory=lambda: {})
    calls: int = 0
    # Semantic-cache verifier requests (LLMVerifier) are answered with this word and
    # counted separately, so `calls` keeps counting real answers only.
    verifier_answer: str = "YES"
    verifier_calls: int = 0

    # -- request handling -------------------------------------------------

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.requests.append({"path": request.url.path, "port": request.url.port, "body": body})
        if self.failures:
            status = self.failures.pop(0)
            return httpx2.Response(
                status,
                json={"error": {"message": f"simulated {status}", "type": "server_error"}},
                headers=self.failure_headers,
            )
        if _is_verifier_request(body):
            self.verifier_calls += 1
            return httpx2.Response(200, json=self._verdict(body))
        self.calls += 1
        path = request.url.path
        if path.endswith("/chat/completions"):
            if body.get("stream"):
                return self._openai_stream(body)
            return httpx2.Response(200, json=self._openai_chat(body))
        if path.endswith("/embeddings"):
            return httpx2.Response(200, json=self._openai_embeddings(body))
        if path.endswith("/messages"):
            if body.get("stream"):
                return self._anthropic_stream(body)
            return httpx2.Response(200, json=self._anthropic_message(body))
        return httpx2.Response(404, json={"error": {"message": "not found", "type": "x"}})

    def _verdict(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": f"verdict-{self.verifier_calls}",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": body.get("model", "gpt-4o-mini"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": self.verifier_answer},
                }
            ],
            "usage": {"prompt_tokens": 150, "completion_tokens": 1, "total_tokens": 151},
        }

    # -- OpenAI payloads ----------------------------------------------------

    def _openai_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": f"{body.get('model', 'gpt-4o')}-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"answer #{self.calls}"},
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        }

    def _openai_stream(self, body: dict[str, Any]) -> httpx2.Response:
        chunks: list[str] = []
        for i, piece in enumerate(["answer ", f"#{self.calls}"]):
            chunk = {
                "id": f"chatcmpl-{self.calls}",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": body.get("model", "gpt-4o"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece}
                        if i
                        else {"role": "assistant", "content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return httpx2.Response(
            200, content="".join(chunks).encode(), headers={"content-type": "text/event-stream"}
        )

    def _openai_embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        raw_input = body.get("input", "")
        inputs: list[Any] = raw_input if isinstance(raw_input, list) else [raw_input]  # pyright: ignore[reportUnknownVariableType]
        return {
            "object": "list",
            "model": body.get("model", "text-embedding-3-small"),
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.1 * (i + 1), 0.2, 0.3]}
                for i in range(len(inputs))
            ],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

    # -- Anthropic payloads -------------------------------------------------

    def _anthropic_message(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": f"msg_{self.calls}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "claude-sonnet-5-5"),
            "content": [{"type": "text", "text": f"answer #{self.calls}"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }

    def _anthropic_stream(self, body: dict[str, Any]) -> httpx2.Response:
        model = body.get("model", "claude-sonnet-5-5")
        events: list[tuple[str, dict[str, Any]]] = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{self.calls}",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1000, "output_tokens": 1},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"answer #{self.calls}"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 500},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        payload = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
        return httpx2.Response(
            200, content=payload.encode(), headers={"content-type": "text/event-stream"}
        )

    # -- client factories ---------------------------------------------------

    def http_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self.handler))

    def async_http_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))


def _is_verifier_request(body: dict[str, Any]) -> bool:
    from caelum.cache.verifier import VERIFIER_PROMPT

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    first = cast("list[Any]", messages)[0]
    if not isinstance(first, dict):
        return False
    return cast("dict[str, Any]", first).get("content") == VERIFIER_PROMPT


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    """A throw-away SQLite cache location (never the user's real ~/.caelum)."""
    return tmp_path / "cache.sqlite"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Make sure no test ever writes into the real user cache directory."""
    monkeypatch.setenv("CAELUM_HOME", str(tmp_path / "caelum-home"))
    yield


@pytest.fixture(autouse=True)
def _cache_at_first_answer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Most tests check what happens once an answer is cached: they cache it at once
    (N=1). Frequency admission (default N=2) has its own tests in test_admission.py."""
    from caelum.cache import admission

    monkeypatch.setattr(admission, "DEFAULT_ADMISSION", 1)
    yield


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Show the false-positive rates measured by test_semantic_pairs.py (task 5)."""
    import sys

    module = sys.modules.get("tests.test_semantic_pairs")
    results = getattr(module, "RESULTS", None) if module is not None else None
    if not results:
        return
    terminalreporter.section("semantic cache: 138 recorded question pairs")
    for rates in results:
        for line in rates.line().splitlines():
            terminalreporter.write_line(line)
