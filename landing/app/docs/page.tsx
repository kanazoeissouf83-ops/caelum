// Static site: no server redirect, so /docs/ points the browser to the first page.
export const metadata = { title: "Docs — CAELUM" };

export default function Page() {
  return (
    <>
      <meta httpEquiv="refresh" content="0; url=/docs/introduction/" />
      <p style={{ padding: 40 }}>
        <a href="/docs/introduction/">Go to the documentation</a>
      </p>
    </>
  );
}
