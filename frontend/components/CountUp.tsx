"use client";

import { useEffect, useRef, useState } from "react";

export function CountUp({
  value,
  decimals = 0,
  suffix = "",
  durationMs = 350,
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  durationMs?: number;
  className?: string;
}) {
  const [display, setDisplay] = useState(0);
  const frame = useRef<number>(0);

  useEffect(() => {
    const start = performance.now();
    const from = 0;
    function tick(now: number) {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - (1 - t) * (1 - t);
      setDisplay(from + (value - from) * eased);
      if (t < 1) frame.current = requestAnimationFrame(tick);
    }
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [value, durationMs]);

  return (
    <span className={`count-up font-mono tabular-nums ${className ?? ""}`}>
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}
