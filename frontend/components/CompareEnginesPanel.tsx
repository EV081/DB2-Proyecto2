import { normalizeResult } from "@/lib/normalize";
import type { App, SearchResponse } from "@/lib/types";
import { CountUp } from "./CountUp";
import { engineColorVar, engineLabel } from "./EngineSelector";
import { LatencyBadge } from "./LatencyBadge";
import { ResultCard } from "./ResultCard";

export interface EngineRun {
  engine: string;
  loading: boolean;
  response?: SearchResponse;
  error?: string;
}

export function CompareEnginesPanel({
  app,
  mediaKind,
  runs,
}: {
  app: App;
  mediaKind: "audio" | "image";
  runs: EngineRun[];
}) {
  const completed = runs.filter((r) => r.response);
  const ranked = [...completed].sort((a, b) => a.response!.latency_ms - b.response!.latency_ms);
  const rankOf = new Map(ranked.map((r, i) => [r.engine, i + 1]));

  return (
    <div className="flex flex-col gap-4">
      {completed.length > 1 && (
        <div className="overflow-hidden rounded-lg border border-border">
          <div className="grid grid-cols-[1.4fr_0.6fr_0.9fr_0.9fr] gap-2 border-b border-border bg-surface px-3 py-2 font-mono text-[10px] uppercase tracking-wide text-muted">
            <span>Motor</span>
            <span>Rank</span>
            <span>Latencia</span>
            <span># resultados</span>
          </div>
          {ranked.map((run) => {
            const color = engineColorVar(run.engine);
            return (
              <div
                key={run.engine}
                className="grid grid-cols-[1.4fr_0.6fr_0.9fr_0.9fr] items-center gap-2 px-3 py-2 font-mono text-xs odd:bg-surface/40"
              >
                <span className="flex items-center gap-2" style={{ color }}>
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
                  {engineLabel(run.engine)}
                </span>
                <span style={rankOf.get(run.engine) === 1 ? { color } : undefined}>
                  P{rankOf.get(run.engine)}
                </span>
                <CountUp value={run.response!.latency_ms} decimals={2} suffix=" ms" />
                <span className="tabular-nums">{run.response!.results.length}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="scanlines grid grid-cols-1 divide-y divide-border overflow-hidden rounded-lg border border-border md:grid-cols-2 md:divide-x md:divide-y-0 xl:grid-cols-4">
        {runs.map((run) => {
          const color = engineColorVar(run.engine);
          const rank = rankOf.get(run.engine);
          const maxScore = run.response?.results[0]?.score;
          return (
            <div key={run.engine} className="flex flex-col gap-3 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ background: color }} />
                  <span className="font-mono text-sm font-semibold uppercase tracking-wide" style={{ color }}>
                    {engineLabel(run.engine)}
                  </span>
                </span>
                {run.response && <LatencyBadge latencyMs={run.response.latency_ms} />}
              </div>
              {rank && (
                <span
                  className="w-fit rounded-full px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wide"
                  style={
                    rank === 1
                      ? { background: `color-mix(in srgb, ${color} 18%, transparent)`, color }
                      : { color: "var(--muted)" }
                  }
                >
                  P{rank} {rank === 1 && "· mas rapido"}
                </span>
              )}
              {run.loading && (
                <p className="flex items-center gap-2 font-mono text-xs text-muted">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: color }} />
                  buscando...
                </p>
              )}
              {run.error && <p className="text-xs text-error">{run.error}</p>}
              {run.response && (
                <div className="flex flex-col gap-2">
                  {run.response.results.length === 0 && (
                    <p className="font-mono text-xs text-muted">Sin resultados</p>
                  )}
                  {run.response.results.slice(0, 3).map((r) => (
                    <ResultCard
                      key={String(r.id)}
                      result={normalizeResult(app, r)}
                      mediaKind={mediaKind}
                      maxScore={maxScore}
                      highlightQuery={run.response!.modality === "text" ? run.response!.query : undefined}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
