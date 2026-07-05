import { CountUp } from "./CountUp";

export function StatsSummary({
  count,
  scores,
  latencyMs,
}: {
  count: number;
  scores: number[];
  latencyMs: number;
}) {
  const top = scores.length ? Math.max(...scores) : 0;
  const bottom = scores.length ? Math.min(...scores) : 0;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      <StatTile label="resultados" color="var(--foreground)">
        <CountUp value={count} decimals={0} />
      </StatTile>
      <StatTile label="latencia" color="var(--engine-spimi)">
        <CountUp value={latencyMs} decimals={2} suffix=" ms" />
      </StatTile>
      {scores.length > 0 && (
        <StatTile label="rango de score" color="var(--engine-pgvector)">
          <CountUp value={top} decimals={3} />
          <span className="text-muted"> – </span>
          <CountUp value={bottom} decimals={3} />
        </StatTile>
      )}
    </div>
  );
}

function StatTile({ label, color, children }: { label: string; color: string; children: React.ReactNode }) {
  return (
    <div
      className="flex flex-col gap-1 rounded-lg border border-border bg-surface px-4 py-3"
      style={{ borderTop: `3px solid ${color}` }}
    >
      <span className="font-mono text-[10px] uppercase tracking-wide text-muted">{label}</span>
      <span className="font-mono text-2xl font-bold tabular-nums" style={{ color }}>
        {children}
      </span>
    </div>
  );
}
