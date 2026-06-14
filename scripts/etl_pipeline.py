import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KAGGLE_BASE = Path("datasets")

DATASET_STRUCTURE = {
    "audio": KAGGLE_BASE / "audio" / "songs",
    "images": KAGGLE_BASE / "images" / "products",
    "text": KAGGLE_BASE / "text" / "lyrics",
}

EXTENSIONS = {
    "audio": (".wav", ".mp3", ".flac"),
    "images": (".jpg", ".jpeg", ".png", ".webp"),
    "text": (".txt", ".json", ".csv"),
}

MODALITY_TAGS = {
    "audio": "audio_text",
    "images": "image_text",
    "text": "audio_text",
}


def find_files(directory: Path, extensions: tuple) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.rglob("*") if p.suffix in extensions])


def scan_datasets() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    print("=" * 60)
    print("  FASE 0 — ESCANEO DE DATASETS")
    print("=" * 60)
    for modality, path in DATASET_STRUCTURE.items():
        files = find_files(path, EXTENSIONS[modality])
        found[modality] = files
        print(f"  [{modality.upper():6}] {path}")
        print(f"         {len(files)} archivos encontrados")
    return found


def phase_extract(data: dict[str, list[Path]]) -> dict[str, dict[str, object]]:
    print("\n" + "=" * 60)
    print("  FASE 1 — EXTRACCIÓN DE CARACTERÍSTICAS")
    print("=" * 60)

    from src.extraction import extract_mfcc_features, extract_sift_features, extract_tfidf_features

    results: dict[str, dict[str, object]] = {"audio": {}, "images": {}, "text": {}}

    for fpath in data["audio"]:
        t0 = time.perf_counter()
        try:
            matrix = extract_mfcc_features(str(fpath))
            elapsed = time.perf_counter() - t0
            results["audio"][fpath.name] = {"path": fpath, "matrix": matrix, "time_s": elapsed}
            print(f"  [AUDIO] {fpath.name:30s} MFCC {list(matrix.shape)}  {elapsed*1000:7.2f} ms")
        except Exception as e:
            print(f"  [AUDIO] {fpath.name:30s} ERROR: {e}")

    for fpath in data["images"]:
        t0 = time.perf_counter()
        try:
            matrix = extract_sift_features(str(fpath))
            elapsed = time.perf_counter() - t0
            results["images"][fpath.name] = {"path": fpath, "matrix": matrix, "time_s": elapsed}
            print(f"  [IMAGE] {fpath.name:30s} SIFT {list(matrix.shape)}  {elapsed*1000:7.2f} ms")
        except Exception as e:
            print(f"  [IMAGE] {fpath.name:30s} ERROR: {e}")

    for fpath in data["text"]:
        t0 = time.perf_counter()
        try:
            matrix = extract_tfidf_features(str(fpath))
            elapsed = time.perf_counter() - t0
            results["text"][fpath.name] = {"path": fpath, "matrix": matrix, "time_s": elapsed}
            print(f"  [TEXT ] {fpath.name:30s} TFIDF {list(matrix.shape)}  {elapsed*1000:7.2f} ms")
        except Exception as e:
            print(f"  [TEXT ] {fpath.name:30s} ERROR: {e}")

    return results


def phase_train_codebook(results: dict, k: int = 50) -> dict:
    print("\n" + "=" * 60)
    print("  FASE 2 — ENTRENAMIENTO DE CODEBOOKS (K-Means)")
    print("=" * 60)

    import numpy as np
    from src.ml.quantizer import KClustering

    codebooks = {}

    for modality in ("audio", "images"):
        matrices = [r["matrix"] for r in results[modality].values()]
        if not matrices:
            codebooks[modality] = None
            print(f"  [{modality.upper()}] Sin datos, se omite codebook")
            continue

        all_vectors = np.vstack(matrices)
        n_vectors = all_vectors.shape[0]
        print(f"  [{modality.upper()}] Total vectores: {n_vectors}, "
              f"dimension: {all_vectors.shape[1]}")

        t0 = time.perf_counter()
        kmeans = KClustering(k_centroids=min(k, n_vectors), clustering_algorithm="kmean", n_init=3)
        labels, centroids = kmeans.clusterize_by_matrix(all_vectors)
        elapsed = time.perf_counter() - t0

        codebook = {"kmeans": kmeans, "centroids": centroids, "k": len(centroids)}
        codebooks[modality] = codebook
        print(f"  [{modality.upper()}] Codebook entrenado: {len(centroids)} centroides "
              f"en {elapsed:.2f}s")

    return codebooks


def phase_quantize(results: dict, codebooks: dict) -> dict:
    print("\n" + "=" * 60)
    print("  FASE 3 — CUANTIZACIÓN Y CONSTRUCCIÓN DE HISTOGRAMAS")
    print("=" * 60)

    import numpy as np

    collections: dict[str, dict[str, dict[str, int]]] = {
        "text": {}, "images": {}, "audio": {}
    }

    for modality in ("audio", "images"):
        codebook = codebooks.get(modality)
        if codebook is None:
            continue
        kmeans = codebook["kmeans"]
        prefix = "a" if modality == "audio" else "v"

        for fname, entry in results[modality].items():
            matrix = entry["matrix"]
            hist: dict[str, int] = {}
            for vec in matrix:
                idx, _ = kmeans.nearest_centroid(vec)
                key = f"{prefix}{idx}"
                hist[key] = hist.get(key, 0) + 1
            collections[modality][fname] = hist
            n_codewords = len(hist)
            print(f"  [{modality.upper()[:5]}] {fname:30s} {n_codewords:3d} codewords activos")

    for fname, entry in results["text"].items():
        matrix = entry["matrix"]
        vec = matrix[0] if matrix.ndim > 1 else matrix
        n_terms = len(vec)
        terms = {f"t{i}": max(1, int(abs(v) * 10 + 1)) for i, v in enumerate(vec[:100])}
        collections["text"][fname] = terms
        print(f"  [TEXT ] {fname:30s} {len(terms):3d} términos")

    return collections


def phase_index_engine(collections: dict) -> tuple:
    print("\n" + "=" * 60)
    print("  FASE 4 — INDEXACIÓN EN MOTOR CUSTOM (TF-IDF)")
    print("=" * 60)

    from src.engine.similarity import build_tfidf_index, cosine_similarity, top_k

    indices = {}
    stats = {}

    for modality, coll in collections.items():
        if not coll:
            indices[modality] = None
            stats[modality] = {"docs": 0, "terms": 0}
            print(f"  [{modality.upper()}] Sin documentos, se omite")
            continue

        t0 = time.perf_counter()
        index, df, n_docs = build_tfidf_index(coll)
        elapsed = time.perf_counter() - t0

        indices[modality] = (index, df, n_docs, coll)
        stats[modality] = {
            "docs": n_docs,
            "terms": len(df),
            "time_s": elapsed,
        }
        print(f"  [{modality.upper()}] {n_docs} docs, {len(df)} términos "
              f"en {elapsed*1000:.1f} ms")

    return indices, stats


def phase_store_db(data: dict, collections: dict) -> dict:
    print("\n" + "=" * 60)
    print("  FASE 5 — ALMACENAMIENTO EN POSTGRESQL")
    print("=" * 60)

    try:
        from src.db.database import SessionLocal
        from src.db.models import Item, Chunk
        from sqlalchemy import text
    except Exception as e:
        print(f"  ERROR importando módulos DB: {e}")
        return {"status": "error", "detail": str(e)}

    inserted_items = 0
    inserted_chunks = 0

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            session.commit()
    except Exception as e:
        print(f"  PostgreSQL no disponible: {e}")
        print("  Los datos quedaron solo en el motor custom (in-memory).")
        return {"status": "skipped", "reason": str(e)}

    for modality, coll in collections.items():
        modality_tag = MODALITY_TAGS.get(modality, modality)
        for doc_id, hist in coll.items():
            try:
                with SessionLocal() as session:
                    clean_id = Path(doc_id).stem
                    item = Item(
                        title=clean_id,
                        modality=modality_tag,
                        meta={"source": str(DATASET_STRUCTURE[modality] / doc_id)},
                    )
                    session.add(item)
                    session.flush()

                    chunk = Chunk(
                        item_id=item.id,
                        chunk_index=0,
                        content=f"Histograma de {modality}: {json.dumps(hist)[:500]}",
                        histogram=hist,
                    )
                    session.add(chunk)
                    session.commit()
                    inserted_items += 1
                    inserted_chunks += 1

            except Exception as e:
                print(f"  [DB ERROR] {doc_id}: {e}")

    print(f"  Items insertados: {inserted_items}")
    print(f"  Chunks insertados: {inserted_chunks}")
    return {"status": "ok", "items": inserted_items, "chunks": inserted_chunks}


def print_pipeline_report(
    scan: dict, results: dict, codebooks: dict,
    indices: dict, index_stats: dict, db_result: dict,
    elapsed: float,
) -> None:
    print("\n" + "=" * 60)
    print("  REPORTE FINAL DEL PIPELINE ETL")
    print("=" * 60)

    total_files = sum(len(files) for files in scan.values())
    processed = sum(len(r) for r in results.values()) if results else 0
    print(f"\n  Archivos escaneados  : {total_files}")
    print(f"  Archivos procesados  : {processed}")
    print(f"  Tiempo total         : {elapsed:.2f}s")

    for modality in ("audio", "images", "text"):
        if modality in index_stats and index_stats[modality]:
            s = index_stats[modality]
            print(f"\n  [{modality.upper()}] Índice TF-IDF:")
            print(f"      Documentos : {s['docs']}")
            print(f"      Términos   : {s['terms']}")
            if s.get("time_s"):
                print(f"      Tiempo idx : {s['time_s']*1000:.1f} ms")

    if codebooks:
        print(f"\n  Codebooks entrenados:")
        for mod, cb in codebooks.items():
            if cb:
                print(f"      {mod}: {cb['k']} centroides")

    print(f"\n  Base de datos:")
    if db_result.get("status") == "ok":
        print(f"      Items  : {db_result.get('items', 0)}")
        print(f"      Chunks : {db_result.get('chunks', 0)}")
    else:
        print(f"      {db_result.get('status', 'error')}: {db_result.get('reason', 'desconocido')}")

    print("\n" + "=" * 60)


def run_pipeline(dataset_base: Path, k_centroids: int = 50, skip_db: bool = False):
    global KAGGLE_BASE, DATASET_STRUCTURE
    KAGGLE_BASE = dataset_base
    DATASET_STRUCTURE = {
        "audio": KAGGLE_BASE / "audio" / "songs",
        "images": KAGGLE_BASE / "images" / "products",
        "text": KAGGLE_BASE / "text" / "lyrics",
    }

    t_start = time.perf_counter()

    data = scan_datasets()
    total_files = sum(len(files) for files in data.values())
    if total_files == 0:
        print("\n  No se encontraron archivos. Pipeline completado (vacio).")
        return

    results = phase_extract(data)
    codebooks = phase_train_codebook(results, k=k_centroids)
    collections = phase_quantize(results, codebooks)
    indices, index_stats = phase_index_engine(collections)

    if skip_db:
        db_result = {"status": "skipped", "reason": "omitido por --skip-db"}
    else:
        db_result = phase_store_db(data, collections)

    elapsed = time.perf_counter() - t_start
    print_pipeline_report(data, results, codebooks, indices, index_stats, db_result, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Maestro ETL — DB2 Proyecto 2")
    parser.add_argument(
        "dataset_path", nargs="?", default=str(KAGGLE_BASE),
        help="Ruta base del directorio datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--k", type=int, default=50,
        help="Número de centroides para K-Means (default: 50)",
    )
    parser.add_argument(
        "--skip-db", action="store_true",
        help="Omitir almacenamiento en PostgreSQL",
    )
    args = parser.parse_args()

    run_pipeline(Path(args.dataset_path), k_centroids=args.k, skip_db=args.skip_db)
