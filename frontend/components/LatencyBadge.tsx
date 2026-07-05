import { CountUp } from "./CountUp";

export function LatencyBadge({ latencyMs }: { latencyMs: number }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-surface px-2.5 py-1 text-xs text-muted">
      <CountUp value={latencyMs} decimals={2} suffix=" ms" />
    </span>
  );
}
