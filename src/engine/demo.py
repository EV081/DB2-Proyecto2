from __future__ import annotations

from src.engine.mock_data import AUSTEN_COLLECTION, COLLECTIONS, QUERIES
from src.engine.similarity import (
    build_tfidf_index,
    chebyshev_distance,
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
    log_tf,
    manhattan_distance,
    top_k,
    vectorize_query,
)


# (metric_name, fn, higher_is_better?)
METRICS = [
    ("cosine",    cosine_similarity,  True),
    ("euclidean", euclidean_distance, False),
    ("manhattan", manhattan_distance, False),
    ("chebyshev", chebyshev_distance, False),
]


def _print_ranking(label: str, ranking: list[tuple[str, float]]) -> None:
    print(f"    [{label}]")
    for doc_id, score in ranking:
        print(f"        {score:8.4f}  {doc_id}")


def _austen_logtf(doc_id: str) -> dict[str, float]:
    return {t: log_tf(tf) for t, tf in AUSTEN_COLLECTION[doc_id].items()}


def demo_austen_from_ppt() -> None:
    print("\n=== EJEMPLO DEL PPT 08 — Novelas de Jane Austen ===")
    print("  (log-TF + L2-normalize, sin idf — tal como el slide)")
    ss = l2_normalize(_austen_logtf("SS_sentido_sensibilidad"))
    op = l2_normalize(_austen_logtf("OP_orgullo_prejuicio"))
    cb = l2_normalize(_austen_logtf("CB_cumbres_borrascosas"))
    print(f"  cos(SS, OP) = {cosine_similarity(ss, op):.4f}   (slide ≈ 0.94)")
    print(f"  cos(SS, CB) = {cosine_similarity(ss, cb):.4f}   (slide ≈ 0.79)")
    print(f"  cos(OP, CB) = {cosine_similarity(op, cb):.4f}   (slide ≈ 0.69)")


def demo_tfidf_pipeline() -> None:
    """Pipeline IR del curso para las 3 modalidades del proyecto."""
    for modality, collection in COLLECTIONS.items():
        index, df, n = build_tfidf_index(collection)
        print(f"\n=== Modalidad: {modality.upper()}  "
              f"({n} docs, {len(df)} términos únicos) ===")

        for q_id, q_tf in QUERIES[modality].items():
            print(f"\n  Query: {q_id}  =>  {q_tf}")
            q_vec = vectorize_query(q_tf, df, n)
            for name, fn, hib in METRICS:
                ranking = top_k(q_vec, index, k=3,
                                score=fn, higher_is_better=hib)
                _print_ranking(name, ranking)


def run() -> None:
    demo_austen_from_ppt()
    demo_tfidf_pipeline()


if __name__ == "__main__":
    run()
