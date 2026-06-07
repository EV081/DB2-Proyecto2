from __future__ import annotations

from math import isclose

from src.engine.mock_data import AUDIO_HISTOGRAMS, AUDIO_QUERIES
from src.engine.similarity import (
    cosine_distance,
    cosine_similarity,
    jaccard_binary,
    jaccard_weighted,
    top_k,
)


def test_cosine_identity() -> None:
    h = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert isclose(cosine_similarity(h, h), 1.0, abs_tol=1e-9)
    assert isclose(cosine_distance(h, h), 0.0, abs_tol=1e-9)


def test_cosine_orthogonal() -> None:
    a = {"x": 1.0}
    b = {"y": 1.0}
    assert cosine_similarity(a, b) == 0.0


def test_cosine_empty_is_zero() -> None:
    assert cosine_similarity({}, {"x": 1.0}) == 0.0
    assert cosine_similarity({}, {}) == 0.0


def test_cosine_known_value() -> None:
    # a·b = 1*2 + 2*1 = 4 ; |a|=sqrt(5) ; |b|=sqrt(5) ; cos = 4/5
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 2.0, "y": 1.0}
    assert isclose(cosine_similarity(a, b), 4 / 5, abs_tol=1e-9)


def test_jaccard_binary_basics() -> None:
    a = {"x": 5, "y": 1}
    b = {"y": 9, "z": 4}
    # |∩|=1 {y}, |∪|=3 {x,y,z} -> 1/3
    assert isclose(jaccard_binary(a, b), 1 / 3, abs_tol=1e-9)
    assert jaccard_binary({}, {}) == 0.0
    assert jaccard_binary(a, a) == 1.0


def test_jaccard_weighted_collapses_to_binary_on_01() -> None:
    a = {"x": 1, "y": 1}
    b = {"y": 1, "z": 1}
    # min:y=1; max:x=1,y=1,z=1 -> 1/3, igual que binario
    assert isclose(jaccard_weighted(a, b), 1 / 3, abs_tol=1e-9)


def test_jaccard_weighted_uses_weights() -> None:
    a = {"x": 4, "y": 2}
    b = {"x": 1, "y": 8}
    # min: 1+2=3 ; max: 4+8=12 -> 0.25
    assert isclose(jaccard_weighted(a, b), 0.25, abs_tol=1e-9)


def test_top_k_returns_k_sorted_desc() -> None:
    q = AUDIO_QUERIES["q_rocky"]
    ranking = top_k(q, AUDIO_HISTOGRAMS, k=3)
    assert len(ranking) == 3
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)


def test_top_k_semantics_rocky_finds_rock() -> None:
    # El primer resultado para una query "rocky" debe ser una canción rock
    # (validación de que los mocks y la métrica están alineados).
    q = AUDIO_QUERIES["q_rocky"]
    best_doc, _ = top_k(q, AUDIO_HISTOGRAMS, k=1)[0]
    assert "rock" in best_doc


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()
