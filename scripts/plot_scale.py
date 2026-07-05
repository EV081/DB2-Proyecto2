from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_COLORS = {
    "spimi":    "#1f77b4",
    "gin":      "#2ca02c",
    "gist":     "#9467bd",
    "pgvector": "#d62728",
}

_SECTIONS = ("music_lyrics", "music_audio", "fashion_desc", "fashion_image")

_METRIC_LABEL = {
    "avg_ms":            "Latencia media (ms)",
    "p95_ms":            "Latencia p95 (ms)",
    "throughput_qps":    "Throughput (qps)",
    "rss_peak_mb":       "RSS pico (MB)",
    "index_size_mb":     "Indice on-disk (MB)",
    "io_seeks_per_query":"Seeks por query (SPIMI)",
}


def _plot_section(section: str, metric: str, reports: list[dict],
                  out_path: Path) -> None:
    # {engine -> [(N, valor), ...]}
    series: dict[str, list[tuple[int, float]]] = {}
    for rep in reports:
        n = rep.get("corpus_size", {}).get(section)
        if n is None or section not in rep:
            continue
        for engine, metrics in rep[section].items():
            v = metrics.get(metric)
            if v is None:
                continue
            series.setdefault(engine, []).append((int(n), float(v)))

    if not series:
        print(f"[skip] {section}: sin datos para metrica '{metric}'")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for engine, pts in series.items():
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2,
                color=_COLORS.get(engine, "#555555"),
                label=engine)
    ax.set_xscale("log")
    ax.set_xlabel("N documentos (log)")
    ax.set_ylabel(_METRIC_LABEL.get(metric, metric))
    ax.set_title(f"Escalabilidad {section} — {_METRIC_LABEL.get(metric, metric)}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok]  {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_jsons", nargs="+", required=True,
                        help="Uno o mas JSON de bench_full (cargas distintas)")
    parser.add_argument("--metric", default="avg_ms",
                        choices=list(_METRIC_LABEL.keys()),
                        help="Metrica a graficar (default: avg_ms)")
    parser.add_argument("--out-dir", default="docs/graphs")
    args = parser.parse_args()

    reports = [json.loads(Path(p).read_text(encoding="utf-8"))
               for p in args.input_jsons]

    out_dir = Path(args.out_dir)
    for section in _SECTIONS:
        filename = f"scale_{section}_{args.metric}.png"
        _plot_section(section, args.metric, reports, out_dir / filename)


if __name__ == "__main__":
    main()