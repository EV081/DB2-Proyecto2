from __future__ import annotations

from math import log10, sqrt
from typing import Callable, Iterable

Vector = dict[str, float]
TermFreqs = dict[str, int]
Collection = dict[str, TermFreqs]


# ===========================================================================
# Pesos: log-TF, IDF, TF-IDF  
# ===========================================================================
def log_tf(tf_raw: int | float) -> float:
    if tf_raw <= 0:
        return 0.0
    return 1.0 + log10(tf_raw)


def document_frequency(collection: Collection) -> dict[str, int]:
    df: dict[str, int] = {}
    for tf_doc in collection.values():
        for term in tf_doc:
            df[term] = df.get(term, 0) + 1
    return df


def idf(df_t: int, n_docs: int) -> float:
    if df_t <= 0 or n_docs <= 0:
        return 0.0
    return log10(n_docs / df_t)


def tfidf_vector(
    tf_doc: TermFreqs,
    df: dict[str, int],
    n_docs: int,
    use_log_tf: bool = True,
) -> Vector:
    weights: Vector = {}
    for term, tf in tf_doc.items():
        wtf = log_tf(tf) if use_log_tf else float(tf)
        if wtf == 0.0:
            continue
        df_t = df.get(term, 0)
        if df_t == 0:
            continue
        weights[term] = wtf * idf(df_t, n_docs)
    return weights


def l2_normalize(v: Vector) -> Vector:
    n = sqrt(sum(w * w for w in v.values()))
    if n == 0.0:
        return dict(v)
    return {k: w / n for k, w in v.items()}


# ===========================================================================
# Producto punto y normas
# ===========================================================================
def dot(a: Vector, b: Vector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(k, 0.0) for k, w in a.items())


def l2_norm(a: Vector) -> float:
    return sqrt(sum(w * w for w in a.values()))


# ===========================================================================
# Coseno
# ===========================================================================
def cosine_similarity(a: Vector, b: Vector) -> float:
    na, nb = l2_norm(a), l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def cosine_distance(a: Vector, b: Vector) -> float:
    return 1.0 - cosine_similarity(a, b)


# ===========================================================================
# 4) Minkowski — Manhattan, Euclidiana, Chebyshev
# ===========================================================================
def minkowski_distance(a: Vector, b: Vector, p: float) -> float:
    keys = set(a) | set(b)
    if p == float("inf"):
        return max((abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), default=0.0)
    if p == 1.0:
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
    s = sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) ** p for k in keys)
    return s ** (1.0 / p)


def manhattan_distance(a: Vector, b: Vector) -> float:
    return minkowski_distance(a, b, p=1.0)


def euclidean_distance(a: Vector, b: Vector) -> float:
    return minkowski_distance(a, b, p=2.0)


def chebyshev_distance(a: Vector, b: Vector) -> float:
    return minkowski_distance(a, b, p=float("inf"))

# ===========================================================================
# Ranking Top-K
# ===========================================================================
ScoreFn = Callable[[Vector, Vector], float]


def top_k(
    query: Vector,
    docs: dict[str, Vector],
    k: int = 5,
    score: ScoreFn = cosine_similarity,
    higher_is_better: bool = True,
) -> list[tuple[str, float]]:
    scored: Iterable[tuple[str, float]] = (
        (doc_id, score(query, vec)) for doc_id, vec in docs.items()
    )
    return sorted(scored, key=lambda x: x[1], reverse=higher_is_better)[:k]


# ===========================================================================
# Helper end-to-end: indexar colección -> consultar query
# ===========================================================================
def build_tfidf_index(
    collection: Collection,
    use_log_tf: bool = True,
    normalize: bool = True,
) -> tuple[dict[str, Vector], dict[str, int], int]:
    df = document_frequency(collection)
    n_docs = len(collection)
    index: dict[str, Vector] = {}
    for doc_id, tf_doc in collection.items():
        v = tfidf_vector(tf_doc, df, n_docs, use_log_tf=use_log_tf)
        index[doc_id] = l2_normalize(v) if normalize else v
    return index, df, n_docs


def vectorize_query(
    tf_query: TermFreqs,
    df: dict[str, int],
    n_docs: int,
    use_log_tf: bool = True,
    normalize: bool = True,
) -> Vector:
    v = tfidf_vector(tf_query, df, n_docs, use_log_tf=use_log_tf)
    return l2_normalize(v) if normalize else v


__all__ = [
    # tipos
    "Vector", "TermFreqs", "Collection", "ScoreFn",
    # pesos
    "log_tf", "document_frequency", "idf",
    "tfidf_vector", "l2_normalize",
    # similitud / distancia
    "dot", "l2_norm",
    "cosine_similarity", "cosine_distance",
    "manhattan_distance", "euclidean_distance",
    "chebyshev_distance", "minkowski_distance",
    # ranking + helpers
    "top_k", "build_tfidf_index", "vectorize_query",
]
