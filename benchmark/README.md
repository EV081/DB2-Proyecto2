# benchmark/

Todo lo necesario para reproducir los benchmarks del INFORME sin tocar la ETL de producción.

## Estrategia

En vez de agregar un flag `--limit` al ETL, se crea un **subset por symlinks** para cada tamaño N y se apunta la ETL a ese subdir. Ventajas:

- La ETL de `scripts/` no cambia.
- El feature cache (`_features/`) se preserva entre corridas — MFCC/SIFT solo se extraen una vez.
- Los subdirs de subset son transitorios y se pueden borrar sin riesgo (`benchmark/tmp/`).

## Layout

```
benchmark/
├── run_all.sh        # orquestador — reindexa a 10k/20k/30k/40k, corre bench + recall + plots
├── README.md
├── results/          # bench_<N>.json, recall_<N>.json por tamaño
├── graphs/           # PNGs: barras del último N + escalabilidad multi-N
└── tmp/              # subdirs con symlinks (transitorios, se recrean por corrida)
```

## Uso

```bash
# Todo de una vez (defaults: 10k, 20k, 30k, 40k)
bash benchmark/run_all.sh

# Tamaños custom
bash benchmark/run_all.sh --sizes 5000,15000,30000

# Regenerar solo los plots sin re-benchear
bash benchmark/run_all.sh --only-plot

# Ajustar carga de queries
bash benchmark/run_all.sh --queries 200 --k 20
```

## Requisitos previos

- Datos completos ya descargados (`bash scripts/setup_all.sh` una vez, con FMA_LIMIT=40000)
- Docker corriendo con Postgres arriba
- matplotlib instalado (`pip install matplotlib`)

## Qué hace cada corrida

Para cada N ∈ {10k, 20k, 30k, 40k}:

1. **Symlinks**: primeros N lyrics/audios/fashion(images+descs) en `tmp/<modality>_<N>/`
2. **Wipe**: `TRUNCATE songs, products, codebooks, search_logs`. Elimina `blocks/` y `final/` de `indexes/` pero **preserva `_features/`** (cache de MFCC/SIFT que es lo caro de recomputar).
3. **ETL music**: reindexa con los subsets. Codebook `1000/500`.
4. **ETL fashion**: reindexa con los subsets. Codebook `1000/1024` + `max-image-samples=150k`.
5. **Bench**: `bench_full.py` con las 100 queries → `results/bench_<N>.json`
6. **Recall**: `compute_recall.py` con 100 queries → `results/recall_<N>.json`

Al terminar los 4 tamaños:
- `plot_bench.py` genera 5 gráficos de barras del último N.
- `plot_scale.py` genera 6 gráficos de escalabilidad (una línea por motor, N en escala log).

## Tiempo estimado

Por N: **~30-60 min** (dominado por K-Means de imagen sobre 10k+ SIFT).
Total 4 sizes: **~2-4 horas**.

Con el feature cache preservado, la segunda corrida en adelante es más rápida (solo re-entrena K-Means + rehace SPIMI + repobla Postgres — no re-extrae SIFT/MFCC).

## Cómo integrar en el INFORME

Los JSONs quedan en `benchmark/results/`. El script `scripts/fill_informe.py` puede leer uno o varios y llenar las tablas de la Sección 4:

```bash
# Rellena INFORME con los 4 tamaños (una fila por N)
for N in 10000 20000 30000 40000; do
    python3 scripts/fill_informe.py \
        --bench benchmark/results/bench_${N}.json \
        --recall benchmark/results/recall_${N}.json \
        --informe docs/INFORME.md
done
```

Los PNGs de `benchmark/graphs/` referenciarse en el INFORME (los paths ya coinciden con `docs/graphs/*.png` — copiá o cambia el path según prefieras).
