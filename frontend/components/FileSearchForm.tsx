"use client";

import { useId, useState } from "react";

export function FileSearchForm({
  accept,
  onSubmit,
  loading,
}: {
  accept: string;
  onSubmit: (file: File) => void;
  loading: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputId = useId();

  function acceptFile(f: File | null) {
    if (f) setFile(f);
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (file) onSubmit(file);
      }}
      className="flex flex-col gap-3 sm:flex-row sm:items-stretch"
    >
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          acceptFile(e.dataTransfer.files?.[0] ?? null);
        }}
        className={`flex flex-1 cursor-pointer items-center gap-3 rounded-md border border-dashed px-4 py-3 text-sm transition-colors ${
          dragging ? "border-engine-spimi bg-engine-spimi/10" : "border-border hover:border-foreground/40"
        }`}
      >
        <svg
          className="h-5 w-5 shrink-0 text-muted"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
          <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
        </svg>
        <span className="min-w-0 truncate text-muted">
          {file ? <span className="text-foreground">{file.name}</span> : "Arrastra un archivo o haz clic para elegir"}
        </span>
        <input
          id={inputId}
          type="file"
          accept={accept}
          onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
          className="hidden"
        />
      </label>
      <button
        type="submit"
        disabled={loading || !file}
        className="rounded-md bg-foreground px-4 py-2 text-sm font-semibold text-background transition-opacity disabled:opacity-40"
      >
        {loading ? "Buscando..." : "Buscar"}
      </button>
    </form>
  );
}
