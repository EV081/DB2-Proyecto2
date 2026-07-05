"use client";

import { useRef, useState } from "react";

function formatTime(t: number): string {
  if (!isFinite(t) || t < 0) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

export function AudioPlayer({ src, color = "var(--foreground)" }: { src: string; color?: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);

  function toggle() {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) audio.pause();
    else void audio.play();
  }

  function seek(e: React.MouseEvent<HTMLDivElement>) {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    audio.currentTime = pct * duration;
    setCurrent(pct * duration);
  }

  const pct = duration ? (current / duration) * 100 : 0;

  return (
    <div className="flex items-center gap-3 rounded-md border border-border bg-surface-2 px-3 py-2.5">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        className="hidden"
      />
      <button
        type="button"
        onClick={toggle}
        aria-label={playing ? "Pausar" : "Reproducir"}
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-background transition-transform active:scale-90"
        style={{ background: color }}
      >
        {playing ? (
          <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
            <rect x="6" y="5" width="4" height="14" rx="1" />
            <rect x="14" y="5" width="4" height="14" rx="1" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5 translate-x-0.5">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        <div onClick={seek} className="h-2 w-full cursor-pointer overflow-hidden rounded-full bg-surface">
          <div
            className="h-full rounded-full transition-[width]"
            style={{ width: `${pct}%`, background: color }}
          />
        </div>
        <div className="flex justify-between font-mono text-[10px] tabular-nums text-muted">
          <span>{formatTime(current)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
    </div>
  );
}
