from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # backend sin display
import matplotlib.pyplot as plt


# Colores estables por motor
_COLORS = {
    "spimi":    "#1f77b4",
    "gin":      "#2ca02c",
    "gist":     "#9467bd",
    "pgvector": "#d62728",
}

# Que graficos generar: (seccion_json, metrica, ylabel, archivo_png)
_PLOTS = [
    ("music_lyrics",  "avg_ms",         "Latencia (ms)",        "music_lyrics_latency.png"),
    ("music_lyrics",  "throughput_qps", "Throughput (qps)",     "music_lyrics_throughput.png"),
    ("music_audio",   "avg_ms",         "Latencia (ms)",        "music_audio_latency.png"),
    ("fashion_desc",  "avg_ms",         "Latencia (ms)",        "fashion_desc_latency.png"),
    ("fashion_image", "avg_ms",         "Latencia (ms)",        "fashion_image_latency.png"),
]


def _bar_chart(section: str, metric: str, ylabel: str, engines_metrics: dict[str, dict],
               out_path: Path, n_queries: int, k: int) -> None:
    engines = list(engines_metrics.keys())
    values = [engines_metrics[e].get(metric, 0.0) for e in engines]
    colors = [_COLORS.get(e, "#555555") for e in engines]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(engines, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{section}   |   n_queries={n_queries}   k={k}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Anota el valor arriba de cada barra
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_json", required=True,
                        help="Path al JSON producido por bench_full.py")
    parser.add_argument("--out-dir", default="docs/graphs",
                        help="Directorio de salida de los PNG")
    args = parser.parse_args()

    report = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    n_queries = report.get("n_queries", 0)
    k = report.get("k", 0)
    out_dir = Path(args.out_dir)

    generados = 0
    for section, metric, ylabel, filename in _PLOTS:
        if section not in report:
            print(f"[skip] seccion '{section}' no esta en el JSON")
            continue
        _bar_chart(section, metric, ylabel, report[section],
                   out_dir / filename, n_queries, k)
        print(f"[ok]  {out_dir / filename}")
        generados += 1

    print(f"\n{generados} graficos generados en {out_dir}")


if __name__ == "__main__":
    main()