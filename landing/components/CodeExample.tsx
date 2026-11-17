import { CodeBlock } from "./CodeBlock";

const BEFORE = `from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)`;

const AFTER = `from caelum import OpenAI   # <- the only change

client = OpenAI(cache=True, retry=True)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
client.metrics.print_summary()  # requests, cache hits, cost`;

export function CodeExample() {
  return (
    <section className="mx-auto max-w-5xl px-5 py-16">
      <h2 className="font-heading text-2xl font-semibold sm:text-3xl">Get started in 30 seconds</h2>
      <div className="mt-6 max-w-md">
        <CodeBlock label="terminal">{`pip install "caelum-ai[openai]"`}</CodeBlock>
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <CodeBlock label="Before">{BEFORE}</CodeBlock>
        <CodeBlock label="After: CAELUM" accent>
          {AFTER}
        </CodeBlock>
      </div>
      <p className="mt-4 text-sm text-paper/60">
        Your code, your types and your API keys stay the same. Caching and retries are opt-in.
        CAELUM never resells tokens: you keep your own provider accounts.
      </p>
    </section>
  );
}
