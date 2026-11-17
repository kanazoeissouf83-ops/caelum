"""See what CAELUM saves you, in one call.

Run (requires an OpenAI API key, costs a fraction of a cent):

    export OPENAI_API_KEY=sk-...
    uv run python examples/metrics_demo.py
"""

from caelum import OpenAI


def main() -> None:
    client = OpenAI(cache=True)  # enable the local exact cache

    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of France?",  # repeated: served from the cache
        "What is the capital of France?",  # repeated again
    ]
    for question in questions:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        print(f"{question} -> {response.choices[0].message.content}")

    print()
    client.metrics.print_summary()
    # Machine-readable version:
    # client.metrics.summary() / client.metrics.export_json("metrics.json")


if __name__ == "__main__":
    main()
