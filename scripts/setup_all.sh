# Orquesta: descarga datasets + levanta Postgres + corre ETLs.
#
# Datasets:
#   - FMA small      
#   - Spotify lyrics 
#   - Fashion images
#
# Flags:
#   --force-download    re-descarga aunque exista .downloaded
#   --force-index       re-indexa aunque exista data/.../final/meta.json
#   --only NAME         corre solo fma | spotify | fashion | music | fashion_etl
#   --no-serve          no levanta el backend FastAPI al final (default: sí lo levanta)
#   --no-etl            solo descarga + levanta Postgres, no corre ETLs (útil antes de benchmark/run_all.sh)
#   --host HOST         host de uvicorn (default: 127.0.0.1)
#   --port PORT         puerto de uvicorn (default: 8000)
#
# Uso típico:
#   bash scripts/setup_all.sh                       # todo + backend
#   bash scripts/setup_all.sh --no-serve            # solo data/indexes
#   bash scripts/setup_all.sh --no-etl --no-serve   # solo baja datasets (para benchmark sweep)
#   bash scripts/setup_all.sh --force-index
#   bash scripts/setup_all.sh --only fma

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Cargar .env si existe (exporta KAGGLE_API_TOKEN, *_DATASET, etc para los hijos)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source ./.env
    set +a
fi

FORCE_DOWNLOAD=0
FORCE_INDEX=0
ONLY=""
SERVE=1
RUN_ETL=1
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-download) FORCE_DOWNLOAD=1; shift ;;
        --force-index)    FORCE_INDEX=1;    shift ;;
        --only)           ONLY="$2";        shift 2 ;;
        --no-serve)       SERVE=0;          shift ;;
        --no-etl)         RUN_ETL=0;        shift ;;
        --host)           SERVE_HOST="$2";  shift 2 ;;
        --port)           SERVE_PORT="$2";  shift 2 ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "flag desconocido: $1" >&2; exit 2 ;;
    esac
done

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

sentinel() { [[ -f "data/$1/.downloaded" ]]; }
mark()     { mkdir -p "data/$1" && touch "data/$1/.downloaded"; }

# index_done <idx_dir> <mod1> [mod2 ...]
# 0 (true) sólo si TODAS las modalidades listadas tienen final/meta.json.
# Asi una corrida anterior que murió a mitad (ej. texto OK pero audio caído)
index_done() {
    local idx_dir="$1"; shift
    [[ $# -gt 0 ]] || return 1
    for mod in "$@"; do
        [[ -f "$idx_dir/$mod/final/meta.json" ]] || return 1
    done
    return 0
}

# FASE 1: descargas
echo "=== FASE 1: descargas ==="

if want fma; then
    if [[ $FORCE_DOWNLOAD -eq 1 ]] || ! sentinel fma; then
        FMA_LIMIT_ARG="${FMA_LIMIT:-40000}"
        echo "[fma] descargando fma_large (~93GB) y extrayendo $FMA_LIMIT_ARG mp3s..."
        python scripts/download_fma.py --out-dir data --limit "$FMA_LIMIT_ARG"
        mark fma
    else
        echo "[fma] ya descargado"
    fi
fi

kaggle_ok() {
    [[ -f "$HOME/.kaggle/kaggle.json" || -f "$HOME/.kaggle/access_token" \
       || -n "${KAGGLE_API_TOKEN:-}" \
       || (-n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}") ]]
}

if want spotify; then
    if ! kaggle_ok; then
        echo "[spotify] SKIP: no hay credenciales Kaggle (~/.kaggle/access_token o equivalentes)"
    elif [[ $FORCE_DOWNLOAD -eq 1 ]] || ! sentinel spotify; then
        echo "[spotify] descargando lyrics..."
        python scripts/download_spotify.py --out-dir data/spotify
        mark spotify
    else
        echo "[spotify] ya descargado"
    fi
fi

if want fashion; then
    if ! kaggle_ok; then
        echo "[fashion] SKIP: no hay credenciales Kaggle"
    elif [[ $FORCE_DOWNLOAD -eq 1 ]] || ! sentinel fashion; then
        echo "[fashion] descargando imágenes..."
        python scripts/download_fashion.py --out-dir data/fashion
        mark fashion
    else
        echo "[fashion] ya descargado"
    fi
fi

# FASE 2: Postgres en Docker
echo "=== FASE 2: Postgres ==="
docker compose up -d
echo -n "Esperando healthcheck"
for i in $(seq 1 30); do
    if docker compose ps 2>/dev/null | grep -q "healthy"; then echo " OK"; break; fi
    echo -n "."; sleep 2
done


# FASE 3: indexar / persistir
if [[ $RUN_ETL -eq 0 ]]; then
    echo "=== FASE 3: saltada (--no-etl) ==="
fi

if [[ $RUN_ETL -eq 1 ]]; then
echo "=== FASE 3: ETL + indexado ==="

# App: música — Spotify (texto) + FMA (audio)
if want music || want spotify || want fma; then
    LYRICS_DIR=""
    AUDIO_DIR=""
    META_ARGS=()
    [[ -d data/spotify/lyrics ]]         && LYRICS_DIR="data/spotify/lyrics"
    [[ -d data/fma_large_flat  ]]        && AUDIO_DIR="data/fma_large_flat"
    [[ -f data/spotify/metadata.csv ]]   && META_ARGS+=(--metadata-csv data/spotify/metadata.csv)
    [[ -f data/fma_large_flat/metadata.csv ]] && META_ARGS+=(--metadata-csv data/fma_large_flat/metadata.csv)

    MUSIC_MODS=()
    [[ -n "$LYRICS_DIR" ]] && MUSIC_MODS+=(text)
    [[ -n "$AUDIO_DIR"  ]] && MUSIC_MODS+=(audio)

    if [[ ${#MUSIC_MODS[@]} -eq 0 ]]; then
        echo "[music] SKIP: no hay lyrics ni audio descargados"
    elif [[ $FORCE_INDEX -eq 1 ]] || ! index_done indexes/music "${MUSIC_MODS[@]}"; then
        # Indice ausente o forzado: ETL completo (codebook + SPIMI + persist)
        python scripts/etl_music.py \
            ${LYRICS_DIR:+--lyrics-dir $LYRICS_DIR} \
            ${AUDIO_DIR:+--audio-dir  $AUDIO_DIR} \
            "${META_ARGS[@]}" \
            --codebook-text 1000 --codebook-audio 500 \
            --index-dir indexes/music \
            --codebooks-dir codebooks \
            --app-name music \
            --reset
    else
        # Indice ya construido: saltar codebook+SPIMI, solo (re)persistir a BD
        echo "[music] indice presente, persistiendo a BD sin reconstruir (--reset)"
        python scripts/persist_only.py \
            --app music \
            ${LYRICS_DIR:+--lyrics-dir $LYRICS_DIR} \
            ${AUDIO_DIR:+--audio-dir  $AUDIO_DIR} \
            "${META_ARGS[@]}" \
            --index-dir indexes/music \
            --codebooks-dir codebooks \
            --reset
    fi
fi

# App: fashion
if want fashion || want fashion_etl; then
    if [[ -d data/fashion/images && -d data/fashion/descs ]]; then
        if [[ $FORCE_INDEX -eq 1 ]] || ! index_done indexes/fashion text image; then
            python scripts/etl_fashion.py \
                --images-dir       data/fashion/images \
                --descriptions-dir data/fashion/descs \
                --metadata-csv     data/fashion/metadata.csv \
                --codebook-image 1024 --codebook-text 1000 \
                --max-image-samples 150000 \
                --index-dir indexes/fashion \
                --codebooks-dir codebooks \
                --app-name fashion \
                --reset
        else
            echo "[fashion] índice ya construido (--force-index para rehacer)"
        fi
    else
        echo "[fashion] SKIP: no hay imágenes/descs descargados"
    fi
fi
fi   # $RUN_ETL

# FASE 4: arrancar backend FastAPI
if [[ $SERVE -eq 1 ]]; then
    echo "=== FASE 4: backend FastAPI ==="
    echo "Arrancando uvicorn en http://${SERVE_HOST}:${SERVE_PORT}  (Ctrl+C para detener)"
    echo "Docs: http://${SERVE_HOST}:${SERVE_PORT}/docs"
    exec uvicorn src.main:app --host "$SERVE_HOST" --port "$SERVE_PORT" --reload
else
    echo "=== Listo (sin levantar API) ==="
    echo "Para arrancar la API manualmente: uvicorn src.main:app --reload"
fi
