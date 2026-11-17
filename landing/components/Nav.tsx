import Link from "next/link";
import { LINKS } from "@/lib";

export function Nav() {
  return (
    <header className="border-b border-navy/60">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-5">
        <Link href="/" className="font-heading text-sm font-semibold tracking-[0.25em]">
          CAELUM
        </Link>
        <nav className="flex items-center gap-6 text-sm text-paper/70">
          <Link href="/pricing" className="hover:text-paper">
            Pricing
          </Link>
          <Link href="/blog" className="hover:text-paper">
            Blog
          </Link>
          <a href={LINKS.docs} className="hover:text-paper">
            Docs
          </a>
          <a href={LINKS.github} className="hover:text-paper">
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
