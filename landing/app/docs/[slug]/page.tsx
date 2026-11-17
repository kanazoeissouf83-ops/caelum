import Link from "next/link";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { docNavigation, docSlugs, readDoc } from "@/docs";

export function generateStaticParams() {
  return docSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = readDoc(slug);
  return { title: `${doc.title} — CAELUM docs`, description: doc.description };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = readDoc(slug);
  const nav = docNavigation();
  const flat = nav.flatMap((g) => g.pages);
  const index = flat.findIndex((p) => p.slug === slug);
  const previous = index > 0 ? flat[index - 1] : null;
  const next = index < flat.length - 1 ? flat[index + 1] : null;
  return (
    <>
      <Nav />
      <div className="mx-auto flex max-w-6xl flex-col gap-10 px-5 py-12 md:flex-row">
        <aside className="md:w-52 md:shrink-0">
          <nav className="flex flex-col gap-6 text-sm md:sticky md:top-8">
            {nav.map((group) => (
              <div key={group.group}>
                <p className="mb-2 font-semibold uppercase tracking-wider text-gold">
                  {group.group}
                </p>
                <ul className="flex flex-col gap-1.5">
                  {group.pages.map((page) => (
                    <li key={page.slug}>
                      <Link
                        href={`/docs/${page.slug}`}
                        className={
                          page.slug === slug ? "text-paper" : "text-paper/60 hover:text-paper"
                        }
                      >
                        {page.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
        </aside>
        <main className="min-w-0 flex-1">
          <h1 className="font-heading text-4xl font-semibold">{doc.title}</h1>
          {doc.description && <p className="mt-3 text-lg text-paper/70">{doc.description}</p>}
          <article className="doc-content mt-8" dangerouslySetInnerHTML={{ __html: doc.html }} />
          <div className="mt-14 flex justify-between gap-4 border-t border-navy/60 pt-6 text-sm">
            {previous ? (
              <Link href={`/docs/${previous.slug}`} className="text-paper/70 hover:text-paper">
                ← {previous.title}
              </Link>
            ) : (
              <span />
            )}
            {next && (
              <Link href={`/docs/${next.slug}`} className="text-gold hover:text-paper">
                {next.title} →
              </Link>
            )}
          </div>
        </main>
      </div>
      <Footer />
    </>
  );
}
