import Link from "next/link";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { postSlugs, readPost } from "@/posts";

export function generateStaticParams() {
  return postSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = readPost(slug);
  return { title: `${post.title} — CAELUM`, description: post.description };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = readPost(slug);
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-3xl px-5 py-16">
        <Link href="/blog" className="text-sm text-paper/60 hover:text-paper">
          ← Blog
        </Link>
        <h1 className="mt-6 font-heading text-4xl font-semibold leading-tight">{post.title}</h1>
        {post.author && <p className="mt-3 text-sm text-paper/60">By {post.author}</p>}
        <article className="doc-content mt-10" dangerouslySetInnerHTML={{ __html: post.html }} />
      </main>
      <Footer />
    </>
  );
}
