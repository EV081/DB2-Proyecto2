"use client";

const STEPS = [5, 10, 20, 50, 100];

export function KSelector({ value, onChange }: { value: number; onChange: (k: number) => void }) {
  const index = Math.max(0, STEPS.indexOf(value) === -1 ? STEPS.indexOf(10) : STEPS.indexOf(value));
  const pct = (index / (STEPS.length - 1)) * 100;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-3">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wide text-muted">Top-k resultados</span>
        <span className="font-display text-2xl font-black text-engine-spimi">{value}</span>
      </div>
      <div className="relative flex items-center py-1">
        <div className="pointer-events-none absolute inset-x-0 h-1.5 rounded-full bg-surface-2">
          <div
            className="h-full rounded-full bg-engine-spimi transition-[width]"
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={0}
          max={STEPS.length - 1}
          step={1}
          value={index}
          onChange={(e) => onChange(STEPS[Number(e.target.value)])}
          className="k-slider relative w-full appearance-none bg-transparent"
        />
      </div>
      <div className="flex justify-between font-mono text-[10px] text-muted">
        {STEPS.map((s) => (
          <span key={s}>{s}</span>
        ))}
      </div>
    </div>
  );
}
