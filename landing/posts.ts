// Blog posts: markdown files in ./posts, rendered at build time.
import fs from "node:fs";
import path from "node:path";
import { renderMarkdown } from "./markdown";

const POSTS_DIR = path.join(process.cwd(), "posts");

export type Post = { slug: string; title: string; description: string; author: string; html: string };

export function postSlugs(): string[] {
  return fs
    .readdirSync(POSTS_DIR)
    .filter((name) => name.endsWith(".md"))
    .map((name) => name.replace(/\.md$/, ""))
    .sort();
}

export function readPost(slug: string): Post {
  const source = fs.readFileSync(path.join(POSTS_DIR, `${slug}.md`), "utf-8");
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  const meta: Record<string, string> = {};
  for (const line of (match?.[1] ?? "").split(/\r?\n/)) {
    const kv = line.match(/^(\w+):\s*"?(.*?)"?\s*$/);
    if (kv) meta[kv[1]] = kv[2];
  }
  const body = match ? source.slice(match[0].length) : source;
  return {
    slug,
    title: meta.title ?? slug,
    description: meta.description ?? "",
    author: meta.author ?? "",
    html: renderMarkdown(body),
  };
}

/** Legal pages: markdown files in ./legal (same format as posts). */
export function readLegal(slug: string): Post {
  const source = fs.readFileSync(path.join(process.cwd(), "legal", `${slug}.md`), "utf-8");
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  const meta: Record<string, string> = {};
  for (const line of (match?.[1] ?? "").split(/\r?\n/)) {
    const kv = line.match(/^(\w+):\s*"?(.*?)"?\s*$/);
    if (kv) meta[kv[1]] = kv[2];
  }
  const body = match ? source.slice(match[0].length) : source;
  return {
    slug,
    title: meta.title ?? slug,
    description: meta.description ?? "",
    author: "",
    html: renderMarkdown(body),
  };
}
