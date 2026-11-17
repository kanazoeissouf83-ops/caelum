"""Post-install smoke test for the published package (no network, no API key).

Run in a clean environment after `pip install 'caelum-ai[openai]'`:

    python ops/smoke_test.py
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import httpx2  # installed with the OpenAI SDK

    import caelum
    from caelum import OpenAI

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "smoke",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with tempfile.TemporaryDirectory() as tmp:
        client = OpenAI(
            api_key="sk-smoke",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
            cache=True,
            cache_admission=1,  # cache on first sight (default is 2: store after a question repeats)
            cache_path=Path(tmp) / "smoke.sqlite",
        )
        messages = [{"role": "user", "content": "ping"}]
        first = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        second = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        assert first.choices[0].message.content == "ok"
        assert caelum.cache_hit(second), "second identical request should be a cache hit"
        client.metrics.print_summary()
        client.close()

    try:
        importlib.import_module("anthropic")
    except ImportError:
        try:
            importlib.import_module("caelum.anthropic")
        except ImportError as error:
            assert "caelum-ai[anthropic]" in str(error), error
            print("missing-extra hint OK:", error)

    print(f"caelum {caelum.__version__} on Python {sys.version.split()[0]}: SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
