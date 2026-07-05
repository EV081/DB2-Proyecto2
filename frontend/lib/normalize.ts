import { mediaUrl } from "./api";
import type { App, MetaField, NormalizedResult, RawResult } from "./types";

/**
 * La descripcion de fashion viene compuesta al momento de la ingesta (ver
 * scripts/download_fashion.py:_compose_desc) como
 * "nombre. genero. categoria. subcategoria. tipo. color. uso. temporada"
 * unidos con ". " y omitiendo los campos vacios. Como category/subcategory
 * ya llegan como columnas propias, los usamos de ancla para separar el
 * resto de campos en vez de asumir posiciones fijas.
 */
const FASHION_TAIL_LABELS = ["Tipo", "Color", "Uso", "Temporada"];

function decomposeFashionDescription(
  description: string | undefined,
  name: string | undefined,
  category: string | undefined,
  subcategory: string | undefined
): MetaField[] | undefined {
  if (!description) return undefined;
  const norm = (s: string) => s.trim().toLowerCase();
  const parts = description
    .split(". ")
    .map((p) => p.trim())
    .filter(Boolean);
  if (parts.length === 0) return undefined;

  const rest = name && norm(parts[0]) === norm(name) ? parts.slice(1) : parts;

  const catIdx = category ? rest.findIndex((p) => norm(p) === norm(category)) : -1;
  const subIdx = subcategory
    ? rest.findIndex((p, i) => (catIdx < 0 || i > catIdx) && norm(p) === norm(subcategory))
    : -1;
  if (catIdx === -1 && subIdx === -1) return undefined;

  const fields: MetaField[] = [];
  if (catIdx > 0) {
    rest.slice(0, catIdx).forEach((value, i) => fields.push({ label: i === 0 ? "Genero" : "Detalle", value }));
  }

  const tailStart = subIdx >= 0 ? subIdx + 1 : catIdx + 1;
  rest.slice(tailStart).forEach((value, i) => {
    fields.push({ label: FASHION_TAIL_LABELS[i] ?? "Detalle", value });
  });

  return fields.length > 0 ? fields : undefined;
}

export function normalizeResult(app: App, raw: RawResult): NormalizedResult {
  const name = raw.title ?? raw.name;
  return {
    id: String(raw.id),
    label: name ?? String(raw.id),
    subtitle: raw.artist ?? raw.category,
    tag: raw.genre ?? raw.subcategory,
    blurb: raw.description,
    metaFields:
      app === "fashion" ? decomposeFashionDescription(raw.description, name, raw.category, raw.subcategory) : undefined,
    score: raw.score,
    engineLabel: raw.engine,
    mediaUrl: mediaUrl(app, raw.id),
    textContent: raw.lyrics_text,
  };
}
