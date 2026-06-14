import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _get_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0


def _peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024
        except Exception:
            pass
        return _get_rss_mb()


def benchmark_custom_engine(
    num_queries: int,
    queries: list[str],
) -> dict:
    print("\n" + "-" * 55)
    print("  Benchmarck: Motor Custom (TF-IDF + Coseno)")
    print("-" * 55)

    from src.engine.mock_data import COLLECTIONS, QUERIES
    from src.engine.similarity import build_tfidf_index, cosine_similarity, top_k, vectorize_query

    available = [m for m, c in COLLECTIONS.items() if c]
    if not available:
        return {"error": "no hay colecciones para benchmark"}

    mod_indexes = {}
    for mod_name in available:
        mod_indexes[mod_name] = build_tfidf_index(COLLECTIONS[mod_name])

    latencies = []
    ram_samples = []
    recall_scores = []

    print(f"  Ejecutando {num_queries} consultas ({len(available)} modalidades)...")

    for qid in range(1, num_queries + 1):
        mod_name = random.choice(available)

        if queries and qid <= len(queries):
            q_text = queries[qid - 1]
            q_tf = {t: 1 for t in q_text.split()}
        else:
            q_index = (qid - 1) % len(QUERIES.get(mod_name, {"default": {"dummy": 1}}))
            q_tf = list(QUERIES[mod_name].values())[q_index]

        index, df, n = mod_indexes.get(mod_name, (None, None, None))
        if index is None:
            continue

        ram_before = _get_rss_mb()
        t0 = time.perf_counter()

        q_vec = vectorize_query(q_tf, df, n)
        ranking = top_k(q_vec, index, k=10, score=cosine_similarity)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ram_after = _get_rss_mb()

        latencies.append(elapsed_ms)
        ram_samples.append(ram_after - ram_before)

        top_scores = [s for _, s in ranking]
        if top_scores:
            recall_scores.append(top_scores[0])

        if qid % max(1, num_queries // 10) == 0:
            print(f"    {qid}/{num_queries}  "
                  f"lat={elapsed_ms:.2f}ms  "
                  f"top={ranking[0][0] if ranking else 'N/A'}")

    if not latencies:
        return {"error": "no data"}

    avg_lat = sum(latencies) / len(latencies)
    peak_ram = max(ram_samples) if ram_samples else 0
    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    result = {
        "avg_latency_ms": round(avg_lat, 2),
        "peak_ram_mb": round(_peak_rss_mb(), 2),
        "recall@10": round(recall, 4),
        "total_queries": len(latencies),
    }

    print(f"\n  Resultados Motor Custom:")
    print(f"    Latencia promedio : {result['avg_latency_ms']:>8.2f} ms")
    print(f"    Pico de RAM       : {result['peak_ram_mb']:>8.2f} MB")
    print(f"    Recall@10         : {result['recall@10']:>8.4f}")

    return result


def benchmark_pgvector(
    num_queries: int,
    queries: list[str],
) -> dict:
    print("\n" + "-" * 55)
    print("  Benchmarck: PostgreSQL + pgvector")
    print("-" * 55)

    try:
        from src.db.database import SessionLocal
        from sqlalchemy import text
    except Exception as e:
        print(f"  ERROR: módulos DB no disponibles: {e}")
        return {"error": str(e)}

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            pgv = session.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar()
            if not pgv:
                print("  WARN: pgvector no está instalado en PostgreSQL")
    except Exception as e:
        print(f"  PostgreSQL no disponible: {e}")
        return {"error": str(e)}

    latencies = []
    ram_samples = []
    recall_scores = []

    print(f"  Ejecutando {num_queries} consultas...")

    for qid in range(1, num_queries + 1):
        ram_before = _get_rss_mb()
        t0 = time.perf_counter()

        try:
            with SessionLocal() as session:
                result = session.execute(
                    text("""
                        SELECT id, title, 1 - (embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]')
                        AS score
                        FROM items
                        ORDER BY embedding <=> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'
                        LIMIT 10
                    """)
                )
                rows = result.fetchall()
        except Exception as e:
            print(f"    [ERROR] consulta {qid}: {e}")
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ram_after = _get_rss_mb()

        latencies.append(elapsed_ms)
        ram_samples.append(ram_after - ram_before)

        if rows:
            recall_scores.append(float(rows[0][2]) if len(rows[0]) > 2 else 0.5)

        if qid % max(1, num_queries // 10) == 0:
            score_str = f"{rows[0][2]:.4f}" if rows else "N/A"
            print(f"    {qid}/{num_queries}  "
                  f"lat={elapsed_ms:.2f}ms  "
                  f"top={score_str}")

    if not latencies:
        return {"error": "no data"}

    avg_lat = sum(latencies) / len(latencies)
    peak_ram = max(ram_samples) if ram_samples else 0
    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    result = {
        "avg_latency_ms": round(avg_lat, 2),
        "peak_ram_mb": round(_peak_rss_mb(), 2),
        "recall@10": round(recall, 4),
        "total_queries": len(latencies),
    }

    print(f"\n  Resultados PostgreSQL + pgvector:")
    print(f"    Latencia promedio : {result['avg_latency_ms']:>8.2f} ms")
    print(f"    Pico de RAM       : {result['peak_ram_mb']:>8.2f} MB")
    print(f"    Recall@10         : {result['recall@10']:>8.4f}")

    return result


def print_report(
    custom_metrics: dict,
    pgvector_metrics: dict,
    output_path: str,
    num_queries: int,
) -> None:
    print("\n" + "=" * 55)
    print("  REPORTE DE BENCHMARK (HITO 2 — FASE 4)")
    print("=" * 55)
    print(f"  Consultas ejecutadas: {num_queries}")

    for engine, metrics in [("custom_engine", custom_metrics),
                              ("pgvector", pgvector_metrics)]:
        print(f"\n  Motor: {engine}")
        if "error" in metrics:
            print(f"    ERROR: {metrics['error']}")
        else:
            print(f"    Latencia promedio : {metrics['avg_latency_ms']:>8.2f} ms")
            print(f"    Pico de RAM       : {metrics['peak_ram_mb']:>8.2f} MB")
            print(f"    Recall@10         : {metrics['recall@10']:>8.4f}")
            print(f"    Consultas OK      : {metrics['total_queries']}")

    print("\n" + "=" * 55)

    full_metrics = {
        "config": {"num_queries": num_queries},
        "custom_engine": custom_metrics,
        "pgvector": pgvector_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  Resultados guardados en: {output_path}")
    print()


def run_benchmark(
    num_queries: int = 100,
    queries_path: str = "",
    output_path: str = "benchmark_results.json",
    skip_pgvector: bool = False,
) -> None:
    queries = []
    if queries_path:
        with open(queries_path) as f:
            queries = [line.strip() for line in f if line.strip()]

    _peak_rss_mb()

    custom_metrics = benchmark_custom_engine(num_queries, queries)

    if skip_pgvector:
        pgvector_metrics = {"error": "omitido por --skip-pgvector"}
    else:
        pgvector_metrics = benchmark_pgvector(num_queries, queries)

    print_report(custom_metrics, pgvector_metrics, output_path, num_queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking Fase 4 — DB2 Proyecto 2"
    )
    parser.add_argument(
        "num_queries", nargs="?", type=int, default=100,
        help="Número de consultas a ejecutar (default: 100)",
    )
    parser.add_argument(
        "--queries", type=str, default="",
        help="Archivo con queries de prueba (una por línea)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Archivo de salida para resultados JSON",
    )
    parser.add_argument(
        "--skip-pgvector", action="store_true",
        help="Omitir benchmark contra PostgreSQL",
    )
    args = parser.parse_args()

    run_benchmark(
        num_queries=args.num_queries,
        queries_path=args.queries,
        output_path=args.output,
        skip_pgvector=args.skip_pgvector,
    )
