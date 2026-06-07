from __future__ import annotations

from src.engine.mock_data import COLLECTIONS, QUERIES
from src.engine.similarity import (
    cosine_similarity,
    jaccard_binary,
    jaccard_weighted,
    top_k,
)


METRICS = {
    "cosine":           cosine_similarity,
    "jaccard_binary":   jaccard_binary,
    "jaccard_weighted": jaccard_weighted,
}


def _print_ranking(title: str, ranking: list[tuple[str, float]]) -> None:
    print(f"  {title}")
    for doc_id, score in ranking:
        print(f"    {score:6.4f}  {doc_id}")


def run() -> None:
    for modality, collection in COLLECTIONS.items():
        print(f"\n=== Modalidad: {modality.upper()}  "
              f"({len(collection)} docs indexados) ===")
        for q_id, q_hist in QUERIES[modality].items():
            print(f"\n  Query: {q_id}  =>  {q_hist}")
            for metric_name, metric_fn in METRICS.items():
                ranking = top_k(q_hist, collection, k=3, metric=metric_fn)
                _print_ranking(f"[{metric_name}] Top-3:", ranking)


if __name__ == "__main__":
    run()
