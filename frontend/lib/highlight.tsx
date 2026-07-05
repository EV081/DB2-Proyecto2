import type { ReactNode } from "react";

/** Resalta las palabras del query dentro de un texto (ej. letra de cancion que hizo match). */
export function highlightText(text: string, query: string | undefined, color: string): ReactNode {
  if (!query || !query.trim()) return text;
  const terms = Array.from(
    new Set(
      query
        .trim()
        .split(/\s+/)
        .filter((t) => t.length > 1)
        .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    )
  ).sort((a, b) => b.length - a.length);
  if (terms.length === 0) return text;

  const re = new RegExp(`(${terms.join("|")})`, "gi");
  const parts = text.split(re);
  const lowerTerms = new Set(terms.map((t) => t.toLowerCase()));

  return parts.map((part, i) =>
    lowerTerms.has(part.toLowerCase()) ? (
      <mark
        key={i}
        className="rounded-sm px-0.5 font-semibold text-foreground"
        style={{ background: `color-mix(in srgb, ${color} 35%, transparent)` }}
      >
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}
