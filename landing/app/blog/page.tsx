import Link from "next/link";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { postSlugs, readPost } from "@/posts";

export const metadata = { title: "Blog — CAELUM" };

export default function Page() {
  const posts = postSlugs().map(readPost);
  return (
    <>
      <Nav />
      <main className="mx-auto min-h-[50vh] max-w-3xl px-5 py-20">
        <h1 className="font-heading text-4xl font-semibold">Blog</h1>
        <ul className="mt-10 flex flex-col gap-8">
          {posts.map((post) => (
            <li key={post.slug} className="border-b border-navy/60 pb-8">
              <Link href={`/blog/${post.slug}`} className="group">
                <h2 className="font-heading text-2xl font-semibold group-hover:text-gold">
                  {post.title}
                </h2>
                <p className="mt-2 text-paper/70">{post.description}</p>
                <p className="mt-3 text-sm text-gold">Read →</p>
              </Link>
            </li>
          ))}
        </ul>
      </main>
      <Footer />
    </>
  );
}
