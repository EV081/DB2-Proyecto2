from __future__ import annotations

from math import isclose, log10, sqrt

from src.engine.similarity import (
    build_tfidf_index,
    chebyshev_distance,
    cosine_distance,
    cosine_similarity,
    document_frequency,
    euclidean_distance,
    idf,
    l2_normalize,
    log_tf,
    manhattan_distance,
    minkowski_distance,
    tfidf_vector,
    top_k,
    vectorize_query,
)

AUSTEN_COLLECTION: dict[str, dict[str, int]] = {
    "SS_sentido_sensibilidad": {"afecto": 115, "celosa": 10, "chisme": 2, "borrascoso": 0},
    "OP_orgullo_prejuicio":    {"afecto": 58,  "celosa": 7,  "chisme": 0, "borrascoso": 0},
    "CB_cumbres_borrascosas":  {"afecto": 20,  "celosa": 11, "chisme": 6, "borrascoso": 38},
}

AUDIO_COLLECTION: dict[str, dict[str, int]] = {
    "song_001_rock":    {"a0": 12, "a1": 5,  "a2": 8,  "a4": 3,  "a7": 1},
    "song_002_rock":    {"a0": 10, "a1": 6,  "a2": 7,  "a4": 4,  "a8": 2},
    "song_003_pop":     {"a1": 4,  "a3": 9,  "a5": 11, "a6": 6,  "a9": 2},
    "song_004_pop":     {"a1": 3,  "a3": 10, "a5": 12, "a6": 5,  "a9": 3},
    "song_005_jazz":    {"a2": 2,  "a4": 1,  "a7": 14, "a8": 9,  "a9": 7},
    "song_006_jazz":    {"a2": 3,  "a4": 2,  "a7": 13, "a8": 10, "a9": 6},
    "song_007_silence": {"a0": 1,  "a9": 1},
}

AUDIO_QUERIES: dict[str, dict[str, int]] = {
    "q_rocky": {"a0": 9,  "a1": 5,  "a2": 7, "a4": 3},
    "q_poppy": {"a3": 8,  "a5": 10, "a6": 5, "a9": 2},
    "q_jazzy": {"a7": 12, "a8": 8,  "a9": 6},
}

#  Pesos: log-TF, IDF, TF-IDF
def test_log_tf_table_from_slide() -> None:
    assert log_tf(0) == 0.0
    assert isclose(log_tf(1), 1.0,  abs_tol=1e-9)
    assert isclose(log_tf(2), 1.3,  abs_tol=1e-2)
    assert isclose(log_tf(10), 2.0, abs_tol=1e-9)
    assert isclose(log_tf(1000), 4.0, abs_tol=1e-9)


def test_idf_calpurnia_example_from_slide() -> None:
    assert isclose(idf(1,      1_000_000), 6.0, abs_tol=1e-9)
    assert isclose(idf(100,    1_000_000), 4.0, abs_tol=1e-9)
    assert isclose(idf(1_000_000, 1_000_000), 0.0, abs_tol=1e-9)


def test_document_frequency_counts_docs_not_occurrences() -> None:
    df = document_frequency(AUSTEN_COLLECTION)
    assert df["afecto"] == 3
    assert df["celosa"] == 3
    assert df["chisme"] == 3
    assert df["borrascoso"] == 3


# Coseno
def _austen_logtf_vector(doc_id: str) -> dict[str, float]:
    return {t: log_tf(tf) for t, tf in AUSTEN_COLLECTION[doc_id].items()}


def test_cosine_matches_slide_values_austen() -> None:
    ss = l2_normalize(_austen_logtf_vector("SS_sentido_sensibilidad"))
    op = l2_normalize(_austen_logtf_vector("OP_orgullo_prejuicio"))
    cb = l2_normalize(_austen_logtf_vector("CB_cumbres_borrascosas"))
    # Tolerancia 0.01 -> coincide con los 2 decimales mostrados en el slide.
    assert isclose(cosine_similarity(ss, op), 0.94, abs_tol=0.01)
    assert isclose(cosine_similarity(ss, cb), 0.79, abs_tol=0.01)
    assert isclose(cosine_similarity(op, cb), 0.69, abs_tol=0.01)

# Coseno propiedades básicas
def test_cosine_identity_and_orthogonal() -> None:
    h = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert isclose(cosine_similarity(h, h), 1.0, abs_tol=1e-9)
    assert isclose(cosine_distance(h, h), 0.0, abs_tol=1e-9)
    assert cosine_similarity({"x": 1.0}, {"y": 1.0}) == 0.0
    assert cosine_similarity({}, {"x": 1.0}) == 0.0


def test_cosine_known_value() -> None:
    # a·b = 4 ; |a|=|b|=sqrt(5) ; cos = 4/5
    assert isclose(cosine_similarity({"x": 1.0, "y": 2.0},
                                     {"x": 2.0, "y": 1.0}),
                   4 / 5, abs_tol=1e-9)


# Minkowski — Manhattan / Euclidiana / Chebyshev
def test_minkowski_family() -> None:
    a = {"x": 1.0, "y": 2.0, "z": 3.0}
    b = {"x": 4.0, "y": 0.0, "z": 0.0}
    # diffs absolutos: |1-4|=3, |2-0|=2, |3-0|=3
    assert isclose(manhattan_distance(a, b), 3 + 2 + 3, abs_tol=1e-9)
    assert isclose(euclidean_distance(a, b), sqrt(9 + 4 + 9), abs_tol=1e-9)
    assert isclose(chebyshev_distance(a, b), 3.0, abs_tol=1e-9)
    # Minkowski p=1 == Manhattan
    assert isclose(minkowski_distance(a, b, 1.0),
                   manhattan_distance(a, b), abs_tol=1e-9)


def test_euclidean_zero_for_identical() -> None:
    a = {"x": 1.0, "y": 2.0}
    assert euclidean_distance(a, a) == 0.0
    assert manhattan_distance(a, a) == 0.0
    assert chebyshev_distance(a, a) == 0.0

# Pipeline end-to-end: TF-IDF + Coseno sobre una colección
def test_tfidf_pipeline_audio_rocky() -> None:
    index, df, n = build_tfidf_index(AUDIO_COLLECTION)
    q = vectorize_query(AUDIO_QUERIES["q_rocky"], df, n)
    ranking = top_k(q, index, k=1, score=cosine_similarity)
    best_doc, _ = ranking[0]
    # El doc más cercano a una query 'rocky' debe ser una canción de rock.
    assert "rock" in best_doc


def test_tfidf_idf_uses_collection_size() -> None:
    # log_tf=False simplifica para chequear que idf entra al peso final.
    df = document_frequency(AUDIO_COLLECTION)
    n = len(AUDIO_COLLECTION)
    v = tfidf_vector({"a0": 1}, df, n, use_log_tf=False)
    # a0 aparece en song_001_rock, song_002_rock, song_007_silence -> df=3
    assert isclose(v["a0"], 1.0 * log10(n / 3), abs_tol=1e-9)


# Runner
def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()