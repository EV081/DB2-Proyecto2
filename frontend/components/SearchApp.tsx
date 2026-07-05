"use client";

import { useEffect, useState } from "react";
import {
  searchFashionDescription,
  searchFashionImage,
  searchMusicAudio,
  searchMusicLyrics,
} from "@/lib/api";
import { normalizeResult } from "@/lib/normalize";
import type { App, FileEngine, SearchResponse, TextEngine } from "@/lib/types";
import { AudioPlayer } from "./AudioPlayer";
import { CompareEnginesPanel, type EngineRun } from "./CompareEnginesPanel";
import { EngineSelector } from "./EngineSelector";
import { FileSearchForm } from "./FileSearchForm";
import { KSelector } from "./KSelector";
import { ResultCard } from "./ResultCard";
import { StatsSummary } from "./StatsSummary";
import { TextSearchForm } from "./TextSearchForm";

const TEXT_ENGINES: TextEngine[] = ["spimi", "gin", "gist"];
const FILE_ENGINES: FileEngine[] = ["spimi", "pgvector"];

const SEARCH_TEXT_BY_APP: Record<App, (query: string, engine: TextEngine, k: number) => Promise<SearchResponse>> = {
  music: searchMusicLyrics,
  fashion: searchFashionDescription,
};

const SEARCH_FILE_BY_APP: Record<App, (file: File, engine: FileEngine, k: number) => Promise<SearchResponse>> = {
  music: searchMusicAudio,
  fashion: searchFashionImage,
};

interface SingleRun {
  loading: boolean;
  response?: SearchResponse;
  error?: string;
}

export interface SearchAppProps {
  app: App;
  mediaKind: "audio" | "image";
  textLabel: string;
  textPlaceholder: string;
  fileLabel: string;
  fileAccept: string;
}

export function SearchApp({
  app,
  mediaKind,
  textLabel,
  textPlaceholder,
  fileLabel,
  fileAccept,
}: SearchAppProps) {
  const searchText = SEARCH_TEXT_BY_APP[app];
  const searchFile = SEARCH_FILE_BY_APP[app];
  const [tab, setTab] = useState<"text" | "file">("text");
  const [textEngine, setTextEngine] = useState<TextEngine>("spimi");
  const [fileEngine, setFileEngine] = useState<FileEngine>("spimi");
  const [compareMode, setCompareMode] = useState(false);
  const [single, setSingle] = useState<SingleRun>({ loading: false });
  const [runs, setRuns] = useState<EngineRun[]>([]);
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryFileUrl, setQueryFileUrl] = useState<string>();
  const [k, setK] = useState(10);

  const engines = tab === "text" ? TEXT_ENGINES : FILE_ENGINES;
  const isLoading = single.loading || runs.some((r) => r.loading);

  useEffect(() => {
    if (!queryFile) {
      setQueryFileUrl(undefined);
      return;
    }
    const url = URL.createObjectURL(queryFile);
    setQueryFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [queryFile]);

  function resetResults() {
    setSingle({ loading: false });
    setRuns([]);
    setQueryFile(null);
  }

  function switchTab(next: "text" | "file") {
    setTab(next);
    resetResults();
  }

  async function runSingle(fn: () => Promise<SearchResponse>) {
    setRuns([]);
    setSingle({ loading: true });
    try {
      const response = await fn();
      setSingle({ loading: false, response });
    } catch (err) {
      setSingle({ loading: false, error: (err as Error).message });
    }
  }

  async function runCompare(fnFor: (engine: string) => () => Promise<SearchResponse>) {
    setSingle({ loading: false });
    setRuns(engines.map((engine) => ({ engine, loading: true })));
    await Promise.all(
      engines.map(async (engine) => {
        try {
          const response = await fnFor(engine)();
          setRuns((prev) => prev.map((r) => (r.engine === engine ? { ...r, loading: false, response } : r)));
        } catch (err) {
          setRuns((prev) =>
            prev.map((r) => (r.engine === engine ? { ...r, loading: false, error: (err as Error).message } : r))
          );
        }
      })
    );
  }

  function handleTextSubmit(query: string) {
    if (compareMode) {
      runCompare((engine) => () => searchText(query, engine as TextEngine, k));
    } else {
      runSingle(() => searchText(query, textEngine, k));
    }
  }

  function handleFileSubmit(file: File) {
    setQueryFile(file);
    if (compareMode) {
      runCompare((engine) => () => searchFile(file, engine as FileEngine, k));
    } else {
      runSingle(() => searchFile(file, fileEngine, k));
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-[1680px] flex-col gap-8 lg:flex-row lg:items-start">
      {/* Sidebar de filtros */}
      <aside className="flex w-full flex-col gap-6 lg:sticky lg:top-24 lg:w-80 lg:shrink-0">
        <nav className="flex rounded-lg border border-border bg-surface p-1">
          <SidebarTab active={tab === "text"} onClick={() => switchTab("text")}>
            {textLabel}
          </SidebarTab>
          <SidebarTab active={tab === "file"} onClick={() => switchTab("file")}>
            {fileLabel}
          </SidebarTab>
        </nav>

        <label className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-sm text-muted">
          <input
            type="checkbox"
            checked={compareMode}
            onChange={(e) => {
              setCompareMode(e.target.checked);
              resetResults();
            }}
          />
          Comparar los {engines.length} motores a la vez
        </label>

        {!compareMode && (
          <div className="flex flex-col gap-2">
            <span className="font-mono text-[11px] uppercase tracking-wide text-muted">Motor</span>
            <EngineSelector
              engines={engines}
              value={tab === "text" ? textEngine : fileEngine}
              onChange={(engine) =>
                tab === "text" ? setTextEngine(engine as TextEngine) : setFileEngine(engine as FileEngine)
              }
            />
          </div>
        )}

        <KSelector value={k} onChange={setK} />

        {tab === "file" && queryFileUrl && (
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3">
            <p className="font-mono text-[11px] uppercase tracking-wide text-muted">Tu búsqueda</p>
            {mediaKind === "audio" ? (
              <AudioPlayer src={queryFileUrl} />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element -- object URL, not a static/remote asset
              <img src={queryFileUrl} alt="Archivo de búsqueda" className="h-40 w-full rounded-md object-cover" />
            )}
          </div>
        )}
      </aside>

      {/* Contenido principal */}
      <div className="flex min-w-0 flex-1 flex-col gap-6">
        {tab === "text" ? (
          <TextSearchForm placeholder={textPlaceholder} onSubmit={handleTextSubmit} loading={isLoading} />
        ) : (
          <FileSearchForm accept={fileAccept} onSubmit={handleFileSubmit} loading={isLoading} />
        )}

        {single.error && <p className="text-sm text-error">{single.error}</p>}

        {single.response && (
          <div className="flex flex-col gap-4">
            <StatsSummary
              count={single.response.results.length}
              scores={single.response.results.map((r) => r.score)}
              latencyMs={single.response.latency_ms}
            />
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
              {single.response.results.map((r) => (
                <ResultCard
                  key={String(r.id)}
                  result={normalizeResult(app, r)}
                  mediaKind={mediaKind}
                  maxScore={single.response!.results[0]?.score}
                  highlightQuery={single.response!.modality === "text" ? single.response!.query : undefined}
                />
              ))}
            </div>
          </div>
        )}

        {runs.length > 0 && <CompareEnginesPanel app={app} mediaKind={mediaKind} runs={runs} />}
      </div>
    </div>
  );
}

function SidebarTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex-1 rounded-md px-3 py-2 text-sm font-semibold transition-colors ${
        active ? "bg-foreground text-background" : "text-muted hover:text-foreground"
      }`}
    >
      {children}
    </button>
  );
}
