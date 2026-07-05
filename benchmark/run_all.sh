#!/usr/bin/env bash
# benchmark/run_all.sh
# Ejecuta el protocolo oficial Fashion 40K para comparar codebooks visuales.
# Varía solo --codebook-image: 128, 512 y 1024.
# Mantiene constantes: dataset 40K, manifest, seed=42, codebook-text=1000,
# max-image-samples=50000 y quality_ks=10,512,1024.
# Este script ejecuta ETL con --reset y luego benchmark completo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZES="10000,20000,30000,40000"
QUERIES=100
K=10
ONLY_PLOT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)     SIZES="$2"; shift 2 ;;
        --queries)   QUERIES="$2"; shift 2 ;;
        --k)         K="$2"; shift 2 ;;
        --only-plot) ONLY_PLOT=1; shift ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "flag desconocido: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -ra SIZE_LIST <<< "$SIZES"

BENCH_DIR="benchmark"
RESULTS_DIR="$BENCH_DIR/results"
GRAPHS_DIR="$BENCH_DIR/graphs"
TMP_DIR="$BENCH_DIR/tmp"

mkdir -p "$RESULTS_DIR" "$GRAPHS_DIR" "$TMP_DIR"

LYRICS_POOL="data/spotify/lyrics"
AUDIO_POOL="data/fma_large_flat"
IMAGES_POOL="data/fashion/images"
DESCS_POOL="data/fashion/descs"

SPOTIFY_META="data/spotify/metadata.csv"
FMA_META="data/fma_large_flat/metadata.csv"
FASHION_META="data/fashion/metadata.csv"

# ---------------------------------------------------------------------------
# Helper: symlink primeros N archivos de src_dir en tgt_dir
# ---------------------------------------------------------------------------
_symlink_subset() {
    local src="$1" tgt="$2" n="$3" pattern="$4"
    rm -rf "$tgt"
    mkdir -p "$tgt"
    if [[ ! -d "$src" ]]; then
        echo "  [warn] no existe $src, salteando" >&2
        return
    fi
    # find + head -N + ln -s. Ordenado por nombre para determinismo.
    # Nota: se separa el listado del loop para evitar SIGPIPE (head cerrando
    # el pipe hace que find/sort exiten con 141 y con pipefail rompe todo).
    local files
    files=$(find "$src" -maxdepth 1 \( -name "$pattern" -type l -o -name "$pattern" -type f \) | sort | head -n "$n" || true)
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        ln -sf "$(realpath "$f")" "$tgt/$(basename "$f")"
    done <<< "$files"
    local got=$(find "$tgt" -maxdepth 1 -name "$pattern" | wc -l)
    echo "  [ok] $tgt: $got archivos"
}


_symlink_fashion_pairs() {
    local n="$1"
    local imgs_tgt="$TMP_DIR/fashion_images_${n}"
    local descs_tgt="$TMP_DIR/fashion_descs_${n}"
    rm -rf "$imgs_tgt" "$descs_tgt"
    mkdir -p "$imgs_tgt" "$descs_tgt"

    python3 - "$IMAGES_POOL" "$DESCS_POOL" "$imgs_tgt" "$descs_tgt" "$n" <<'PY'
import os, sys
from pathlib import Path
imgs_dir, descs_dir, imgs_tgt, descs_tgt, n = sys.argv[1:]
n = int(n)
imgs = {p.stem: p for p in Path(imgs_dir).glob("*.jpg")}
descs = {p.stem: p for p in Path(descs_dir).glob("*.txt")}
common = sorted(set(imgs) & set(descs))[:n]
for stem in common:
    (Path(imgs_tgt) / f"{stem}.jpg").symlink_to(imgs[stem].resolve())
    (Path(descs_tgt) / f"{stem}.txt").symlink_to(descs[stem].resolve())
print(f"  [ok] fashion pairs: {len(common)}")
PY
}

# ---------------------------------------------------------------------------
# Wipe entre corridas: mata SPIMI + codebooks, PRESERVA feature cache
# ---------------------------------------------------------------------------
_wipe_indexes_keep_features() {
    # Elimina blocks/ y final/ pero no _features/
    find indexes -maxdepth 3 -type d \( -name blocks -o -name final \) \
        -exec rm -rf {} + 2>/dev/null || true
    rm -rf codebooks
}

# ---------------------------------------------------------------------------
# TRUNCATE songs/products/codebooks/search_logs en Postgres
# ---------------------------------------------------------------------------
_truncate_db() {
    docker exec db2_proyecto2_postgres psql -U postgres -d proyecto2 -c "
        TRUNCATE songs, products, codebooks, search_logs RESTART IDENTITY CASCADE;
        ANALYZE songs; ANALYZE products;
    " >/dev/null
    echo "  [ok] Postgres: songs/products/codebooks truncados"
}

# ---------------------------------------------------------------------------
# Corrida de un tamanio
# ---------------------------------------------------------------------------
run_size() {
    local N="$1"
    echo
    echo "===================================================="
    echo "  N = $N"
    echo "===================================================="

    echo "[1/5] Symlinks para subset de $N archivos"
    _symlink_subset "$LYRICS_POOL" "$TMP_DIR/lyrics_${N}" "$N" "*.txt"
    _symlink_subset "$AUDIO_POOL"  "$TMP_DIR/audio_${N}"  "$N" "*.mp3"
    _symlink_fashion_pairs "$N"

   
    echo "[2/5] Wipe DB + SPIMI (features preservados)"
    _truncate_db
    _wipe_indexes_keep_features

    echo "[3/5] ETL music"
    META_ARGS=()
    [[ -f "$SPOTIFY_META" ]] && META_ARGS+=(--metadata-csv "$SPOTIFY_META")
    [[ -f "$FMA_META"     ]] && META_ARGS+=(--metadata-csv "$FMA_META")
    python scripts/etl_music.py \
        --lyrics-dir "$TMP_DIR/lyrics_${N}" \
        --audio-dir  "$TMP_DIR/audio_${N}" \
        "${META_ARGS[@]}" \
        --codebook-text 1000 --codebook-audio 500 \
        --index-dir indexes/music \
        --codebooks-dir codebooks \
        --app-name music \
        --reset

    echo "[4/5] ETL fashion"
    FASHION_META_ARG=()
    [[ -f "$FASHION_META" ]] && FASHION_META_ARG=(--metadata-csv "$FASHION_META")
    python scripts/etl_fashion.py \
        --images-dir       "$TMP_DIR/fashion_images_${N}" \
        --descriptions-dir "$TMP_DIR/fashion_descs_${N}" \
        "${FASHION_META_ARG[@]}" \
        --codebook-image 1024 --codebook-text 1000 \
        --max-image-samples 150000 \
        --index-dir indexes/fashion \
        --codebooks-dir codebooks \
        --app-name fashion \
        --reset

    echo "[5/5] Bench + recall"
    python3 scripts/bench_full.py \
        --queries "$QUERIES" --k "$K" \
        --out-json "$RESULTS_DIR/bench_${N}.json" \
        --out-md   "$RESULTS_DIR/bench_${N}.md"
    python3 scripts/compute_recall.py \
        --queries "$QUERIES" --k "$K" \
        --out-json "$RESULTS_DIR/recall_${N}.json"

    echo "N=$N terminado"
}

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
if [[ $ONLY_PLOT -eq 0 ]]; then
    for N in "${SIZE_LIST[@]}"; do
        run_size "$N"
    done
fi

# ---------------------------------------------------------------------------
# Plots: bar chart del ultimo N + escalabilidad sobre todos los N
# ---------------------------------------------------------------------------
LAST_JSON="$RESULTS_DIR/bench_${SIZE_LIST[-1]}.json"
if [[ -f "$LAST_JSON" ]]; then
    echo
    echo "=== Plots bar chart (ultimo N) ==="
    python3 scripts/plot_bench.py --in "$LAST_JSON" --out-dir "$GRAPHS_DIR"
fi

BENCH_JSONS=("$RESULTS_DIR"/bench_*.json)
if [[ ${#BENCH_JSONS[@]} -gt 1 ]]; then
    echo
    echo "=== Plots de escalabilidad (multi-N) ==="
    for METRIC in avg_ms p95_ms throughput_qps rss_peak_mb index_size_mb io_seeks_per_query; do
        python3 scripts/plot_scale.py \
            --in "${BENCH_JSONS[@]}" \
            --metric "$METRIC" \
            --out-dir "$GRAPHS_DIR"
    done
fi

echo
echo "Listo."
echo "  JSONs:  $RESULTS_DIR/"
echo "  Plots:  $GRAPHS_DIR/"
