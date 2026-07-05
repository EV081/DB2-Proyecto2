"use client";

import { useEffect, useState } from "react";
import { fetchDbStatus } from "@/lib/api";

type Status = "checking" | "ok" | "down";

export function SystemStatus() {
  const [status, setStatus] = useState<Status>("checking");
  const [detail, setDetail] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    fetchDbStatus()
      .then((res) => {
        if (cancelled) return;
        setStatus("ok");
        setDetail(`postgres ${res.database} · pgvector ${res.pgvector}`);
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("down");
        setDetail("backend no disponible");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const color = status === "ok" ? "var(--success)" : status === "down" ? "var(--error)" : "var(--muted)";

  return (
    <span className="flex items-center gap-1.5 font-mono text-[11px] text-muted" title={detail}>
      <span
        className={`h-1.5 w-1.5 rounded-full ${status === "checking" ? "animate-pulse" : ""}`}
        style={{ background: color }}
      />
      {status === "checking" && "verificando..."}
      {status === "ok" && "sistema en linea"}
      {status === "down" && "sistema caido"}
    </span>
  );
}
