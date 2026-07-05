export type App = "music" | "fashion";

export type TextEngine = "spimi" | "gin" | "gist";
export type FileEngine = "spimi" | "pgvector";

export interface RawResult {
  id: string | number;
  title?: string;
  name?: string;
  artist?: string;
  category?: string;
  subcategory?: string;
  genre?: string;
  description?: string;
  lyrics_text?: string;
  score: number;
  engine: string;
}

export interface SearchResponse {
  app: App;
  modality: "text" | "audio" | "image";
  engine: string;
  query: string;
  k: number;
  latency_ms: number;
  results: RawResult[];
}

export interface MetaField {
  label: string;
  value: string;
}

export interface NormalizedResult {
  id: string;
  label: string;
  subtitle?: string;
  tag?: string;
  blurb?: string;
  /** Campos de metadata desglosados desde la descripcion compuesta (ver normalize.ts), ej. fashion. */
  metaFields?: MetaField[];
  score: number;
  engineLabel: string;
  mediaUrl: string;
  textContent?: string;
}
