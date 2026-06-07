from __future__ import annotations

from math import sqrt
from typing import Callable, Iterable

Histogram = dict[str, float]
SimilarityFn = Callable[[Histogram, Histogram], float]


# ---------------------------------------------------------------------------
# Coseno
# ---------------------------------------------------------------------------
def _dot(a: Histogram, b: Histogram) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(key, 0.0) for key, weight in a.items())


def _norm(a: Histogram) -> float:
    return sqrt(sum(w * w for w in a.values()))


def cosine_similarity(a: Histogram, b: Histogram) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def cosine_distance(a: Histogram, b: Histogram) -> float:
    return 1.0 - cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------
def jaccard_binary(a: Histogram, b: Histogram) -> float:

    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def jaccard_weighted(a: Histogram, b: Histogram) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = 0.0
    den = 0.0
    for k in keys:
        x = a.get(k, 0.0)
        y = b.get(k, 0.0)
        num += x if x < y else y
        den += x if x > y else y
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Ranking Top-K (preview Hito 2: se combinará con el inverted index)
# ---------------------------------------------------------------------------
def top_k(
    query: Histogram,
    collection: dict[str, Histogram],
    k: int = 5,
    metric: SimilarityFn = cosine_similarity,
) -> list[tuple[str, float]]:
    scored: Iterable[tuple[str, float]] = (
        (doc_id, metric(query, hist)) for doc_id, hist in collection.items()
    )
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


__all__ = [
    "Histogram",
    "SimilarityFn",
    "cosine_similarity",
    "cosine_distance",
    "jaccard_binary",
    "jaccard_weighted",
    "top_k",
]
