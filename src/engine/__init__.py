"""Motor de búsqueda customizado (Capa 3 — dueño: Elmer).

Hito 1: matemática IR del curso — log-TF, IDF, TF-IDF, Coseno y
familia Minkowski (Manhattan/Euclidiana/Chebyshev).
El índice invertido SPIMI llega en Hito 2.
"""

from src.engine.similarity import (
    # tipos
    Collection,
    ScoreFn,
    TermFreqs,
    Vector,
    # pesos
    build_tfidf_index,
    document_frequency,
    idf,
    l2_normalize,
    log_tf,
    tfidf_vector,
    vectorize_query,
    # similitud
    cosine_distance,
    cosine_similarity,
    dot,
    l2_norm,
    # distancias Minkowski
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    # ranking
    top_k,
)

__all__ = [
    "Collection", "ScoreFn", "TermFreqs", "Vector",
    "build_tfidf_index", "document_frequency", "idf",
    "l2_normalize", "log_tf", "tfidf_vector", "vectorize_query",
    "cosine_distance", "cosine_similarity", "dot", "l2_norm",
    "chebyshev_distance", "euclidean_distance",
    "manhattan_distance", "minkowski_distance",
    "top_k",
]
