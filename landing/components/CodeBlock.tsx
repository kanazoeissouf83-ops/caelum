export function CodeBlock({
  label,
  children,
  accent = false,
}: {
  label: string;
  children: string;
  accent?: boolean;
}) {
  return (
    <div className={`rounded-lg border ${accent ? "border-gold/60" : "border-navy"} bg-[#0a1422]`}>
      <div
        className={`border-b px-4 py-2 font-mono text-xs ${
          accent ? "border-gold/40 text-gold" : "border-navy text-paper/50"
        }`}
      >
        {label}
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-paper/90">
        <code>{children}</code>
      </pre>
    </div>
  );
}
