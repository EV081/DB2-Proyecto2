"use client";

import { useState } from "react";
import { highlightText } from "@/lib/highlight";
import type { NormalizedResult } from "@/lib/types";
import { AudioPlayer } from "./AudioPlayer";
import { CountUp } from "./CountUp";
import { engineColorVar, engineDisplayLabel } from "./EngineSelector";

export function ResultCard({
  result,
  mediaKind,
  maxScore,
  highlightQuery,
}: {
  result: NormalizedResult;
  mediaKind: "image" | "audio";
  /** Score del mejor resultado de esta misma lista — dibuja la barra de "gap" relativo. */
  maxScore?: number;
  /** Query de la busqueda por texto — resalta las palabras que dieron match dentro de la letra. */
  highlightQuery?: string;
}) {
  const color = engineColorVar(result.engineLabel);
  const barPct = maxScore && maxScore > 0 ? Math.max(4, (result.score / maxScore) * 100) : 100;
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface p-4 pl-5 transition-transform hover:-translate-y-0.5"
      style={{ borderLeft: `4px solid ${color}` }}
    >
      {mediaKind === "image" ? (
        // eslint-disable-next-line @next/next/no-img-element -- media URL is dynamic and server-side, not worth Next/Image optimization
        <img
          src={result.mediaUrl}
          alt={result.label}
          className="h-44 w-full rounded-md bg-surface-2 object-cover"
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.visibility = "hidden";
          }}
        />
      ) : result.textContent ? (
        <div className="h-44 w-full overflow-y-auto rounded-md bg-surface-2 p-2.5 font-mono text-xs leading-relaxed whitespace-pre-line">
          {highlightText(result.textContent, highlightQuery, color)}
        </div>
      ) : (
        <AudioPlayer src={result.mediaUrl} color={color} />
      )}
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="font-display truncate text-base font-bold">{result.label}</p>
          {result.subtitle && <p className="truncate text-xs text-muted">{result.subtitle}</p>}
        </div>
        <CountUp value={result.score} decimals={3} className="shrink-0 text-sm font-semibold" />
      </div>
      {result.tag && (
        <span className="w-fit rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-muted">
          {result.tag}
        </span>
      )}
      {result.metaFields && result.metaFields.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <dl className="flex flex-wrap gap-x-3 gap-y-1">
            {result.metaFields.map((f, i) => (
              <div key={i} className="flex items-baseline gap-1">
                <dt className="font-mono text-[10px] uppercase tracking-wide text-muted">{f.label}</dt>
                <dd className="text-xs">{highlightText(f.value, highlightQuery, color)}</dd>
              </div>
            ))}
          </dl>
          {result.blurb && (
            <>
              <button
                type="button"
                onClick={() => setExpanded((v) => !v)}
                className="w-fit text-xs font-semibold underline decoration-dotted underline-offset-2 hover:text-foreground"
                style={{ color }}
              >
                {expanded ? "Ocultar texto original" : "Ver texto original"}
              </button>
              {expanded && <p className="text-xs text-muted">{highlightText(result.blurb, highlightQuery, color)}</p>}
            </>
          )}
        </div>
      ) : (
        result.blurb && (
          <div className="flex flex-col gap-1">
            <p className={expanded ? "text-xs text-muted" : "line-clamp-2 text-xs text-muted"}>
              {highlightText(result.blurb, highlightQuery, color)}
            </p>
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="w-fit text-xs font-semibold underline decoration-dotted underline-offset-2 hover:text-foreground"
              style={{ color }}
            >
              {expanded ? "Ver menos" : "Ver descripcion completa"}
            </button>
          </div>
        )
      )}
      <div className="h-1 w-full overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${barPct}%`, background: color }}
        />
      </div>
      <span className="font-mono text-[11px] uppercase tracking-wide" style={{ color }}>
        {engineDisplayLabel(result.engineLabel)}
      </span>
    </div>
  );
}
