import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { LINKS } from "@/lib";

export const metadata = { title: "Pricing — CAELUM" };

type Plan = {
  name: string;
  price: string;
  note: string;
  features: string[];
  cta: { label: string; href?: string };
  highlight?: boolean;
};

const PLANS: Plan[] = [
  {
    name: "Open-source SDK",
    price: "Free",
    note: "Forever, no limits · Apache 2.0",
    features: [
      "Drop-in for the OpenAI, Anthropic, Gemini and Mistral Python SDKs",
      "Exact cache and verified semantic cache (local SQLite)",
      "Retries, multi-provider fallback, routing by config",
      "Local metrics: requests, cache hits, cost saved and spent",
      "Runs entirely on your machine: nothing is sent to CAELUM",
    ],
    cta: { label: "Get started", href: "/docs/quick-start" },
    highlight: true,
  },
  {
    name: "CAELUM Cloud Starter",
    price: "$29",
    note: "per 30 days · 17,500 FCFA · prepaid, card or Mobile Money, no automatic charge",
    features: [
      "1,000,000 questions per month: one cache shared by all your servers and serverless functions",
      "Repetitions counted across all your servers, so answers are reused sooner",
      "Cost per end customer: know what each user costs you",
      "Budget alerts by email before a surprise bill",
      "Dashboard with 90 days of history",
      "Free Cloud plan to try it: 10,000 questions per month, dashboard with 7 days of history",
    ],
    cta: { label: "Start free", href: "https://app.caelum-ai.com/sign-up" },
  },
  {
    name: "Enterprise",
    price: "Custom",
    note: "Annual contract",
    features: [
      "Deployment in your own cloud (VPC)",
      "Governance, audit trails and compliance support",
      "Dedicated support and SLA",
    ],
    cta: { label: "Contact us (soon)" },
  },
];

export default function Page() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-5xl px-5 py-20">
        <h1 className="font-heading text-4xl font-semibold">Pricing</h1>
        <p className="mt-4 max-w-2xl text-paper/70">
          The SDK is free and open source, and stays that way. You keep your own provider
          accounts and pay OpenAI, Anthropic or Google directly: CAELUM never resells tokens.
        </p>
        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {PLANS.map((plan) => (
            <section
              key={plan.name}
              className={`flex flex-col rounded-xl border p-6 ${
                plan.highlight ? "border-gold/70 bg-navy/25" : "border-navy/70"
              }`}
            >
              <h2 className="font-heading text-lg font-semibold">{plan.name}</h2>
              <p className="mt-3 font-heading text-3xl font-semibold text-gold">{plan.price}</p>
              <p className="mt-1 text-sm text-paper/60">{plan.note}</p>
              <ul className="mt-6 flex flex-1 flex-col gap-2.5 text-sm text-paper/80">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex gap-2">
                    <span className="text-gold">✓</span>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
              {plan.cta.href ? (
                <a
                  href={plan.cta.href}
                  className={`mt-8 rounded-lg px-4 py-2.5 text-center text-sm font-medium ${
                    plan.highlight
                      ? "bg-gold text-night hover:bg-paper"
                      : "border border-navy text-paper hover:border-gold"
                  }`}
                >
                  {plan.cta.label}
                </a>
              ) : (
                <span className="mt-8 rounded-lg border border-navy/60 px-4 py-2.5 text-center text-sm text-paper/50">
                  {plan.cta.label}
                </span>
              )}
            </section>
          ))}
        </div>
        <p className="mt-10 text-sm text-paper/50">
          Enterprise prices are set per contract. Questions?{" "}
          <a href={LINKS.github} className="text-gold underline underline-offset-2">
            Open a discussion on GitHub
          </a>
          .
        </p>
      </main>
      <Footer />
    </>
  );
}
