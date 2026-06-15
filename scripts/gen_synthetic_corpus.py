from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def generate_corpus(
    n_docs: int,
    out_path: Path,
    vocab_size: int = 5000,
    min_doc_len: int = 50,
    max_doc_len: int = 200,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    terms_pool = [f"t_{i:05d}" for i in range(vocab_size)]
    weights = [1.0 / (i + 1) for i in range(vocab_size)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    total_postings = 0

    with out_path.open("w", encoding="utf-8") as f:
        for doc_idx in range(n_docs):
            doc_len = rng.randint(min_doc_len, max_doc_len)
            sampled = rng.choices(terms_pool, weights=weights, k=doc_len)
            tf: dict[str, int] = {}
            for t in sampled:
                tf[t] = tf.get(t, 0) + 1
            doc_id = f"doc_{doc_idx:06d}"
            f.write(json.dumps({"id": doc_id, "tf": tf}, ensure_ascii=False))
            f.write("\n")
            total_postings += len(tf)

    elapsed = time.perf_counter() - t0
    stats = {
        "n_docs": n_docs,
        "vocab_size": vocab_size,
        "total_unique_postings": total_postings,
        "avg_terms_per_doc": round(total_postings / n_docs, 1),
        "file_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
        "generation_seconds": round(elapsed, 2),
        "seed": seed,
    }
    return stats


def iter_corpus(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["id"], obj["tf"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de corpus sintetico para SPIMI")
    parser.add_argument("--n", type=int, required=True,
                        help="Numero de documentos a generar")
    parser.add_argument("--vocab", type=int, default=5000,
                        help="Tamano del vocabulario (default 5000)")
    parser.add_argument("--min-len", type=int, default=50,
                        help="Min terminos por doc")
    parser.add_argument("--max-len", type=int, default=200,
                        help="Max terminos por doc")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--out", type=str, required=True,
                        help="Ruta del archivo JSONL de salida")
    args = parser.parse_args()

    stats = generate_corpus(
        n_docs=args.n,
        out_path=Path(args.out),
        vocab_size=args.vocab,
        min_doc_len=args.min_len,
        max_doc_len=args.max_len,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))