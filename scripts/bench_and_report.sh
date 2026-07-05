set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUERIES=100
K=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queries) QUERIES="$2"; shift 2 ;;
        --k)       K="$2";       shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "flag desconocido: $1" >&2; exit 2 ;;
    esac
done

BENCH_JSON="benchmark_fase4.json"
BENCH_MD="benchmark_fase4.md"
RECALL_JSON="recall_fase4.json"

echo "=== [1/4] Benchmark (latencia, throughput) ==="
python3 scripts/bench_full.py \
    --queries "$QUERIES" --k "$K" \
    --out-json "$BENCH_JSON" --out-md "$BENCH_MD"

echo
echo "=== [2/4] Recall@K ==="
python3 scripts/compute_recall.py \
    --queries "$QUERIES" --k "$K" \
    --out-json "$RECALL_JSON"

echo
echo "=== [3/4] Graficos PNG ==="
python3 scripts/plot_bench.py --in "$BENCH_JSON" --out-dir docs/graphs

SCALE_JSONS=(benchmark_fase4_*.json)
if [[ ${#SCALE_JSONS[@]} -gt 1 && -f "${SCALE_JSONS[0]}" ]]; then
    echo "--- Escalabilidad (multi-carga) ---"
    for METRIC in avg_ms p95_ms throughput_qps rss_peak_mb index_size_mb; do
        python3 scripts/plot_scale.py --in "${SCALE_JSONS[@]}" \
            --metric "$METRIC" --out-dir docs/graphs
    done
fi

echo
echo "=== [4/4] Rellenando INFORME ==="
python3 scripts/fill_informe.py \
    --bench "$BENCH_JSON" \
    --recall "$RECALL_JSON" \
    --informe docs/INFORME.md

echo
echo "Listo."
echo "  JSON bench:  $BENCH_JSON"
echo "  JSON recall: $RECALL_JSON"
echo "  Graficos:    benchmark/graphs/*.png"
echo "  Informe:     docs/INFORME.md"