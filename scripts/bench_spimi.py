from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gen_synthetic_corpus import iter_corpus
from src.engine.inverted_index import (
    InvertedIndex,
    build_meta,
    merge_blocks,
    spimi_invert,
)
from src.engine.similarity import (
    build_tfidf_index,
    cosine_similarity,
    top_k,
    vectorize_query,
)


def _rss_mb() -> float:
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
            return 0.0
        return 0.0


def _peak_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return _rss_mb()


def sample_queries(vocab_size: int, n_queries: int, seed: int) -> list[dict[str, int]]:
    rng = random.Random(seed)
    terms = [f"t_{i:05d}" for i in range(vocab_size)]
    weights = [1.0 / (i + 1) for i in range(vocab_size)]
    queries = []
    for _ in range(n_queries):
        q_len = rng.randint(1, 5)
        sampled = rng.choices(terms, weights=weights, k=q_len)
        tf: dict[str, int] = {}
        for t in sampled:
            tf[t] = tf.get(t, 0) + 1
        queries.append(tf)
    return queries


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bench_size(
    corpus_path: Path,
    index_dir: Path,
    n_queries: int,
    block_size_postings: int,
    vocab_size: int,
    query_seed: int,
) -> dict:
    if index_dir.exists():
        shutil.rmtree(index_dir)

    n_docs = sum(1 for _ in iter_corpus(corpus_path))
    print(f"\n--- corpus: {corpus_path.name}  ({n_docs} docs) ---")

    # SPIMI build
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    blocks = spimi_invert(
        iter_corpus(corpus_path),
        block_size_postings=block_size_postings,
        out_dir=index_dir / "blocks",
    )
    postings_path, vocab_path = merge_blocks(blocks, index_dir / "final")
    meta_path = build_meta(postings_path, vocab_path, index_dir / "final")
    spimi_build_s = time.perf_counter() - t0
    spimi_build_ram = _peak_rss_mb() - rss_before
    print(f"  SPIMI build:  {spimi_build_s:7.2f} s   peak RAM +{spimi_build_ram:.1f} MB")

    # Dense build (carga toda la coleccion a memoria)
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    collection = dict(iter_corpus(corpus_path))
    dense_index, df, n = build_tfidf_index(collection)
    dense_build_s = time.perf_counter() - t0
    dense_build_ram = _peak_rss_mb() - rss_before
    print(f"  Dense build:  {dense_build_s:7.2f} s   peak RAM +{dense_build_ram:.1f} MB")

    # Query benchmark
    queries = sample_queries(vocab_size, n_queries, query_seed)

    # SPIMI queries
    latencies_spimi = []
    with InvertedIndex(postings_path, vocab_path, meta_path) as idx:
        for q in queries:
            t0 = time.perf_counter()
            idx.search_topk(q, k=10)
            latencies_spimi.append((time.perf_counter() - t0) * 1000)
        io_seeks_total = idx.io_seeks
        io_bytes_total = idx.io_read_bytes

    # Dense queries
    latencies_dense = []
    for q in queries:
        t0 = time.perf_counter()
        q_vec = vectorize_query(q, df, n)
        top_k(q_vec, dense_index, k=10, score=cosine_similarity)
        latencies_dense.append((time.perf_counter() - t0) * 1000)

    print(f"  SPIMI query:  avg={statistics.mean(latencies_spimi):6.3f} ms"
          f"   p50={percentile(latencies_spimi,50):6.3f}"
          f"   p95={percentile(latencies_spimi,95):6.3f}"
          f"   io_seeks={io_seeks_total}")
    print(f"  Dense query:  avg={statistics.mean(latencies_dense):6.3f} ms"
          f"   p50={percentile(latencies_dense,50):6.3f}"
          f"   p95={percentile(latencies_dense,95):6.3f}")

    return {
        "corpus": corpus_path.name,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "build": {
            "spimi_seconds": round(spimi_build_s, 3),
            "spimi_peak_ram_mb": round(spimi_build_ram, 2),
            "dense_seconds": round(dense_build_s, 3),
            "dense_peak_ram_mb": round(dense_build_ram, 2),
            "block_size_postings": block_size_postings,
            "num_blocks": len(blocks),
        },
        "query": {
            "spimi": {
                "avg_ms": round(statistics.mean(latencies_spimi), 4),
                "p50_ms": round(percentile(latencies_spimi, 50), 4),
                "p95_ms": round(percentile(latencies_spimi, 95), 4),
                "io_seeks_total": io_seeks_total,
                "io_read_bytes_total": io_bytes_total,
                "io_seeks_per_query": round(io_seeks_total / n_queries, 2),
            },
            "dense": {
                "avg_ms": round(statistics.mean(latencies_dense), 4),
                "p50_ms": round(percentile(latencies_dense, 50), 4),
                "p95_ms": round(percentile(latencies_dense, 95), 4),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SPIMI vs denso")
    parser.add_argument("--corpora", nargs="+", default=[
        "data/synthetic_1k.jsonl",
        "data/synthetic_10k.jsonl",
        "data/synthetic_100k.jsonl",
    ])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=500_000)
    parser.add_argument("--vocab", type=int, default=5000)
    parser.add_argument("--query-seed", type=int, default=7)
    parser.add_argument("--index-dir", type=str, default="data/spimi_index")
    parser.add_argument("--out", type=str, default="benchmark_spimi.json")
    args = parser.parse_args()

    results = []
    for corpus in args.corpora:
        p = Path(corpus)
        if not p.exists():
            print(f"WARN: {p} no existe — saltando")
            continue
        r = bench_size(
            corpus_path=p,
            index_dir=Path(args.index_dir) / p.stem,
            n_queries=args.queries,
            block_size_postings=args.block_size,
            vocab_size=args.vocab,
            query_seed=args.query_seed,
        )
        results.append(r)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()