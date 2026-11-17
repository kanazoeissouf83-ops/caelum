import { LINKS } from "@/lib";

export function Hero() {
  return (
    <section className="mx-auto max-w-5xl px-5 pb-16 pt-20 sm:pt-28">
      <p className="mb-5 font-mono text-xs uppercase tracking-widest text-gold">
        Open source · Apache 2.0 · Python SDK
      </p>
      <h1 className="font-heading max-w-3xl text-4xl font-semibold leading-tight tracking-tight sm:text-6xl">
        The control plane for AI inference
      </h1>
      <p className="mt-6 max-w-2xl text-lg leading-relaxed text-paper/75">
        Drop-in replacement for the OpenAI, Anthropic, Google Gemini and Mistral Python SDKs.
        Cache, route, observe and fall back across providers, by changing one import.
      </p>
      <div className="mt-9 flex flex-wrap gap-3">
        <a
          href={LINKS.docs}
          className="rounded-md bg-gold px-5 py-2.5 text-sm font-semibold text-night hover:brightness-110"
        >
          Get started
        </a>
        <a
          href={LINKS.github}
          className="rounded-md border border-navy px-5 py-2.5 text-sm font-semibold hover:border-gold hover:text-gold"
        >
          View on GitHub
        </a>
      </div>
    </section>
  );
}
