"use client";

import { useState } from "react";

export function TextSearchForm({
  placeholder,
  onSubmit,
  loading,
}: {
  placeholder: string;
  onSubmit: (query: string) => void;
  loading: boolean;
}) {
  const [query, setQuery] = useState("");

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (query.trim()) onSubmit(query.trim());
      }}
      className="flex gap-2"
    >
      <div className="relative flex-1">
        <svg
          className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <circle cx="11" cy="11" r="7" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          className="w-full rounded-md border border-border bg-background py-2 pr-3 pl-9 text-sm outline-none transition-shadow focus:border-engine-spimi focus:ring-2 focus:ring-engine-spimi/20"
        />
      </div>
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="rounded-md bg-foreground px-4 py-2 text-sm font-semibold text-background transition-opacity disabled:opacity-40"
      >
        {loading ? "Buscando..." : "Buscar"}
      </button>
    </form>
  );
}
