"use client";

const ENGINE_LABELS: Record<string, string> = {
  spimi: "SPIMI (propio)",
  gin: "Postgres GIN",
  gist: "Postgres GiST",
  pgvector: "pgvector (HNSW)",
};

export function engineLabel(engine: string): string {
  return ENGINE_LABELS[engine] ?? engine;
}

const ENGINE_HINTS: Record<string, string> = {
  spimi: "Indice invertido propio + TF-IDF/histogramas, similitud coseno.",
  gin: "Indice invertido generalizado sobre tsvector, full-text exacto.",
  gist: "Arbol balanceado con firma comprimida, full-text con verificacion.",
  pgvector: "ANN sobre vectores densos con grafo HNSW, similitud coseno aproximada.",
};

export function engineHint(engine: string): string {
  return ENGINE_HINTS[engine] ?? "";
}

const ENGINE_COLOR_VAR: Record<string, string> = {
  spimi: "--engine-spimi",
  gin: "--engine-gin",
  gist: "--engine-gist",
  pgvector: "--engine-pgvector",
};

/** Normaliza el string crudo del backend (ej. "postgres_gin_full_text", "pgvector_hnsw_cosine") a la key de motor. */
export function normalizeEngineKey(rawEngine: string): keyof typeof ENGINE_COLOR_VAR {
  const e = rawEngine.toLowerCase();
  if (e.includes("gist")) return "gist";
  if (e.includes("gin")) return "gin";
  if (e.includes("pgvector")) return "pgvector";
  return "spimi";
}

export function engineColorVar(engine: string): string {
  const key = ENGINE_COLOR_VAR[engine] ? engine : normalizeEngineKey(engine);
  return `var(${ENGINE_COLOR_VAR[key] ?? "--muted"})`;
}

/** Etiqueta legible para el string crudo que devuelve el backend (ej. "postgres_gin_full_text" -> "Postgres GIN"). */
export function engineDisplayLabel(rawEngine: string): string {
  const key = ENGINE_COLOR_VAR[rawEngine] ? rawEngine : normalizeEngineKey(rawEngine);
  return engineLabel(key);
}

export function EngineSelector({
  engines,
  value,
  onChange,
}: {
  engines: readonly string[];
  value: string;
  onChange: (engine: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {engines.map((engine) => {
        const active = value === engine;
        const color = engineColorVar(engine);
        return (
          <button
            key={engine}
            type="button"
            onClick={() => onChange(engine)}
            style={
              active
                ? { borderColor: color, background: `color-mix(in srgb, ${color} 10%, transparent)` }
                : undefined
            }
            className={`flex flex-col gap-0.5 rounded-md border px-3 py-2.5 text-left transition-colors ${
              active ? "" : "border-transparent hover:bg-surface"
            }`}
          >
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
              <span className="font-mono text-sm font-semibold" style={active ? { color } : undefined}>
                {engineLabel(engine)}
              </span>
            </span>
            <span className="pl-4 text-xs leading-snug text-muted">{engineHint(engine)}</span>
          </button>
        );
      })}
    </div>
  );
}
