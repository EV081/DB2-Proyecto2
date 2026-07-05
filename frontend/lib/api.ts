import type { App, FileEngine, SearchResponse, TextEngine } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function extractError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail ?? `Error ${res.status}`;
  } catch {
    return `Error ${res.status}`;
  }
}

function buildUrl(path: string, params: Record<string, string>): string {
  const url = new URL(path, API_URL);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}

async function searchText(
  path: string,
  query: string,
  engine: TextEngine,
  k: number
): Promise<SearchResponse> {
  const res = await fetch(buildUrl(path, { q: query, engine, k: String(k) }));
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

async function searchFile(
  path: string,
  file: File,
  engine: FileEngine,
  k: number
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(buildUrl(path, { engine, k: String(k) }), {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

export function searchMusicLyrics(query: string, engine: TextEngine, k = 10) {
  return searchText("/api/music/search/lyrics", query, engine, k);
}

export function searchMusicAudio(file: File, engine: FileEngine, k = 10) {
  return searchFile("/api/music/search/audio", file, engine, k);
}

export function searchFashionDescription(query: string, engine: TextEngine, k = 10) {
  return searchText("/api/fashion/search/description", query, engine, k);
}

export function searchFashionImage(file: File, engine: FileEngine, k = 10) {
  return searchFile("/api/fashion/search/image", file, engine, k);
}

export function mediaUrl(app: App, id: string | number): string {
  return buildUrl(`/api/${app}/media/${id}`, {});
}

export async function fetchDbStatus(): Promise<{ database: string; pgvector: string }> {
  const res = await fetch(buildUrl("/api/db/status", {}));
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}
