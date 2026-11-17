// Reads the documentation pages (../docs/*.mdx, also used by Mintlify) at build time and
// turns them into HTML, so the docs are readable on the site itself.
import fs from "node:fs";
import path from "node:path";
import { renderMarkdown } from "./markdown";

const DOCS_DIR = path.join(process.cwd(), "..", "docs");

type Nav = { group: string; pages: string[] }[];

export type DocPage = {
  slug: string;
  title: string;
  description: string;
  html: string;
};

export function docNavigation(): { group: string; pages: { slug: string; title: string }[] }[] {
  const config = JSON.parse(fs.readFileSync(path.join(DOCS_DIR, "docs.json"), "utf-8"));
  const groups: Nav = config.navigation.groups;
  return groups.map((g) => ({
    group: g.group,
    pages: g.pages.map((slug) => ({ slug, title: readDoc(slug).title })),
  }));
}

export function docSlugs(): string[] {
  return docNavigation().flatMap((g) => g.pages.map((p) => p.slug));
}

function frontmatter(source: string): { meta: Record<string, string>; body: string } {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!match) return { meta: {}, body: source };
  const meta: Record<string, string> = {};
  for (const line of match[1].split(/\r?\n/)) {
    const kv = line.match(/^(\w+):\s*"?(.*?)"?\s*$/);
    if (kv) meta[kv[1]] = kv[2];
  }
  return { meta, body: source.slice(match[0].length) };
}

export function readDoc(slug: string): DocPage {
  const source = fs.readFileSync(path.join(DOCS_DIR, `${slug}.mdx`), "utf-8");
  const { meta, body } = frontmatter(source);
  const prepared = body
    // Mintlify callouts -> plain HTML blocks (markdown inside is still rendered).
    .replace(/<(Note|Warning)>/g, (_, kind: string) => `<div class="callout callout-${kind.toLowerCase()}">\n\n`)
    .replace(/<\/(Note|Warning)>/g, "\n\n</div>")
    // Internal links: "/caching" -> "/docs/caching".
    .replace(/\]\(\/([a-z-]+)(#[^)]*)?\)/g, (_, page: string, hash = "") => `](/docs/${page}${hash})`);
  return {
    slug,
    title: meta.title ?? slug,
    description: meta.description ?? "",
    html: renderMarkdown(prepared),
  };
}
