import { notFound } from "next/navigation";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { readLegal } from "@/posts";

const PAGES = ["terms", "privacy"] as const;

export function generateStaticParams() {
  return PAGES.map((legal) => ({ legal }));
}

export const dynamicParams = false;

export async function generateMetadata({ params }: { params: Promise<{ legal: string }> }) {
  const { legal } = await params;
  const page = readLegal(legal);
  return { title: `${page.title} — CAELUM`, description: page.description };
}

export default async function Page({ params }: { params: Promise<{ legal: string }> }) {
  const { legal } = await params;
  if (!(PAGES as readonly string[]).includes(legal)) notFound();
  const page = readLegal(legal);
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-5 py-16">
        <h1 className="font-heading text-4xl font-semibold">{page.title}</h1>
        <article className="doc-content mt-8" dangerouslySetInnerHTML={{ __html: page.html }} />
      </main>
      <Footer />
    </>
  );
}
