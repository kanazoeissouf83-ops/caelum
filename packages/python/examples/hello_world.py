"""Hello world: CAELUM as a drop-in replacement for the OpenAI SDK.

Run (requires an OpenAI API key, costs a fraction of a cent):

    export OPENAI_API_KEY=sk-...
    uv run python examples/hello_world.py
"""

from caelum import OpenAI  # the only change compared to: from openai import OpenAI


def main() -> None:
    client = OpenAI()  # reads OPENAI_API_KEY from the environment, like the SDK

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    )
    print(response.choices[0].message.content)

    # Streaming works exactly like with the OpenAI SDK.
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
