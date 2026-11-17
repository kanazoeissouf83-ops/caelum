import { LINKS } from "@/lib";

export function Footer() {
  return (
    <footer className="border-t border-navy/60">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-5 py-8 text-sm text-paper/55">
        <span>© 2026 Issouf KANAZOE · Apache 2.0</span>
        <nav className="flex gap-5">
          <a href={LINKS.docs} className="hover:text-paper">
            Docs
          </a>
          <a href={LINKS.github} className="hover:text-paper">
            GitHub
          </a>
          <a href="https://x.com/caelum_ai73q7m" className="hover:text-paper">
            X (@caelum_ai73q7m)
          </a>
          <a href="/pricing" className="hover:text-paper">
            Pricing
          </a>
          <a href="/blog" className="hover:text-paper">
            Blog
          </a>
          <a href="/terms" className="hover:text-paper">
            Terms
          </a>
          <a href="/privacy" className="hover:text-paper">
            Privacy
          </a>
        </nav>
      </div>
    </footer>
  );
}
