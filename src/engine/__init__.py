"""Motor de búsqueda customizado (Capa 3 — dueño: Elmer).

Hito 1: matemática de similitud (Coseno/Jaccard) sobre histogramas
hardcoded. El índice invertido SPIMI llega en Hito 2.
"""

from src.engine.similarity import (
    Histogram,
    SimilarityFn,
    cosine_distance,
    cosine_similarity,
    jaccard_binary,
    jaccard_weighted,
    top_k,
)

__all__ = [
    "Histogram",
    "SimilarityFn",
    "cosine_similarity",
    "cosine_distance",
    "jaccard_binary",
    "jaccard_weighted",
    "top_k",
]
