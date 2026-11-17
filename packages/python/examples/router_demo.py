"""Send each request to the cheapest capable model, with automatic fallback.

Run (requires OpenAI and Anthropic API keys, costs a fraction of a cent):

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra yaml python examples/router_demo.py
"""

from pathlib import Path

import caelum
from caelum import OpenAI


def main() -> None:
    client = OpenAI(
        router_config=Path(__file__).with_name("router.yaml"),
        # If OpenAI is down, answer with Claude instead (needs ANTHROPIC_API_KEY).
        fallback=["anthropic:claude-sonnet-5-5"],
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Rewrite the user's text in plain English."},
            {"role": "user", "content": "Utilize the aforementioned methodology henceforth."},
        ],
    )
    # Same OpenAI response object, even when Claude answered.
    print(response.choices[0].message.content)
    print("answered by:", caelum.routed_to(response) or response.model)

    client.metrics.print_summary()


if __name__ == "__main__":
    main()
