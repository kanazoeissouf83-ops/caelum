/** @type {import('next').NextConfig} */
// WHY output "export": the site is fully static (docs, blog and legal pages are rendered at
// build time), so it is served as plain files by Cloudflare (Workers static assets).
const nextConfig = { reactStrictMode: true, output: "export", trailingSlash: true };
export default nextConfig;
