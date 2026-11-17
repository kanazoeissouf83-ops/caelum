// Call to action for CAELUM Cloud (it used to be a waitlist before the Cloud opened; the
// component keeps its name so the home page import does not change).
export function Waitlist() {
  return (
    <section id="cloud" className="mx-auto max-w-5xl px-5 py-16">
      <div className="rounded-lg border border-navy bg-navy/20 p-8">
        <h2 className="font-heading text-2xl font-semibold">CAELUM Cloud is open</h2>
        <p className="mt-2 max-w-xl text-sm text-paper/70">
          One cache shared by all your servers, the cost of each of your customers, and budget
          alerts before a surprise bill. Free up to 10,000 questions a month; Starter is $29
          (17,500 FCFA) every 30 days, card or Mobile Money, nothing charged automatically.
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <a
            href="https://app.caelum-ai.com/sign-up"
            className="rounded-md bg-gold px-4 py-2 text-sm font-semibold text-night hover:brightness-110"
          >
            Start free
          </a>
          <a href="/pricing" className="rounded-md border border-navy px-4 py-2 text-sm hover:border-gold">
            See pricing
          </a>
        </div>
      </div>
    </section>
  );
}
