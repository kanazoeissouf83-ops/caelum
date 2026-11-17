import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";

/**
 * Render trusted Markdown to sanitized HTML.
 *
 * The blog, docs and legal pages are authored in this repo and rendered at build time, so
 * there is no untrusted input today. Sanitizing anyway is defense in depth: if any of this
 * content ever comes from elsewhere (an external contribution, a CMS), a <script> or an
 * onerror handler in the Markdown cannot reach a visitor's browser.
 */
export function renderMarkdown(body: string): string {
  const html = marked.parse(body, { async: false }) as string;
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
}
