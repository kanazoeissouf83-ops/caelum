"""CAELUM as a drop-in replacement for the Anthropic SDK.

Run (requires an Anthropic API key, costs a fraction of a cent):

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python examples/anthropic_demo.py
"""

from caelum import Anthropic  # the only change compared to: from anthropic import Anthropic


def main() -> None:
    client = Anthropic(cache=True)  # reads ANTHROPIC_API_KEY, enables the local cache

    for _ in range(2):  # the second identical request is served from the cache
        message = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system="You answer in one short sentence.",
            messages=[{"role": "user", "content": "What is an inference cache?"}],
        )
        block = message.content[0]
        print(block.text if block.type == "text" else block)

    # Streaming helpers are forwarded untouched to the Anthropic SDK.
    with client.messages.stream(
        model="claude-haiku-5-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()

    client.metrics.print_summary()


if __name__ == "__main__":
    main()
