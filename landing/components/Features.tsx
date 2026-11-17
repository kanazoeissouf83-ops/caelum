const FEATURES = [
  {
    icon: "◈",
    title: "Exact & semantic cache",
    body: "Identical requests are answered locally from SQLite. An opt-in semantic cache reuses answers to rephrased questions, after a small model confirms both ask the same thing.",
  },
  {
    icon: "⇄",
    title: "Routing by config",
    body: "Send requests to different models from a dict, JSON or YAML file. OpenAI ⇄ Anthropic request translation built in.",
  },
  {
    icon: "◉",
    title: "Built-in observability",
    body: "client.metrics tracks calls, tokens, cache hits, latency and estimated cost. Export to JSON.",
  },
  {
    icon: "↻",
    title: "Multi-provider fallback",
    body: "When a provider fails, retry with backoff, then fall back to the model you chose on another provider, or to a free local model (Ollama, llama.cpp, vLLM, LM Studio).",
  },
];

const STATS = [
  { value: "30%", label: "of calls saved on a benchmark with 30% repeated requests" },
  { value: "< 1 ms", label: "p95 overhead measured for the exact cache (target: under 5 ms)" },
  { value: "0 / 50", label: "look-alike questions answered wrongly by the semantic cache, with 37 of 50 rephrasings reused (real API test)" },
  { value: "4", label: "SDKs wrapped: OpenAI, Anthropic, Google Gemini, Mistral, plus local models" },
];

export function Features() {
  return (
    <section className="mx-auto max-w-5xl px-5 py-16">
      <h2 className="font-heading text-2xl font-semibold sm:text-3xl">Why CAELUM</h2>
      <div className="mt-8 grid gap-px overflow-hidden rounded-lg border border-navy bg-navy sm:grid-cols-2">
        {FEATURES.map((f) => (
          <div key={f.title} className="bg-night p-6">
            <div className="mb-3 text-xl text-gold" aria-hidden>
              {f.icon}
            </div>
            <h3 className="font-heading font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-paper/70">{f.body}</p>
          </div>
        ))}
      </div>
      <dl className="mt-10 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((s) => (
          <div key={s.value}>
            <dt className="font-heading text-3xl font-semibold text-gold">{s.value}</dt>
            <dd className="mt-1 text-sm text-paper/65">{s.label}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-6 text-xs text-paper/45">
        Figures come from our own benchmarks (see the repository). Your savings depend on how often
        your prompts repeat.
      </p>
    </section>
  );
}
