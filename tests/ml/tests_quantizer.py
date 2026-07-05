from __future__ import annotations

import numpy as np

from src.ml.quantizer import VectorQuantizer, WordQuantizer

# VectorQuantizer
def _vq_1d() -> VectorQuantizer:
    return VectorQuantizer(np.array([[0.0], [5.0], [10.0]]))


def test_nearest_centroid_returns_closest() -> None:
    vq = _vq_1d()
    idx, vec = vq.nearest_centroid(np.array([4.0]))
    # 4 esta mas cerca de 5 que de 0 o 10
    assert idx == 1
    assert float(vec[0]) == 5.0


def test_nearest_centroid_handles_boundary() -> None:
    vq = _vq_1d()
    # 2.5 equidista de 0 y 5; argmin devuelve el primero
    idx, _ = vq.nearest_centroid(np.array([2.5]))
    assert idx == 0


def test_histogram_counts_assignments() -> None:
    vq = _vq_1d()
    matrix = np.array([[1.0], [2.0], [6.0], [9.0], [11.0]])
    hist = vq.histogram(matrix)
    # 1,2 -> cluster 0; 6 -> cluster 1; 9,11 -> cluster 2
    assert hist.tolist() == [2, 1, 2]


def test_histogram_returns_zeros_for_empty_input() -> None:
    vq = _vq_1d()
    hist = vq.histogram(np.empty((0, 1)))
    assert hist.tolist() == [0, 0, 0]


# WordQuantizer
def test_word_histogram_counts_bag_terms() -> None:
    wq = WordQuantizer(["hola", "mundo", "test", "foo"])
    hist = wq.histogram(["hola", "hola", "foo", "bar", "test"])
    # hola=2, mundo=0, test=1, foo=1; "bar" no esta en el vocab -> ignorado
    assert hist.tolist() == [2, 0, 1, 1]


def test_word_histogram_ignores_out_of_vocab_tokens() -> None:
    wq = WordQuantizer(["a", "b"])
    hist = wq.histogram(["x", "y", "z"])
    assert hist.tolist() == [0, 0]


def test_word_histogram_empty_tokens() -> None:
    wq = WordQuantizer(["a", "b", "c"])
    hist = wq.histogram([])
    assert hist.tolist() == [0, 0, 0]


if __name__ == "__main__":
    test_nearest_centroid_returns_closest()
    test_nearest_centroid_handles_boundary()
    test_histogram_counts_assignments()
    test_histogram_returns_zeros_for_empty_input()
    test_word_histogram_counts_bag_terms()
    test_word_histogram_ignores_out_of_vocab_tokens()
    test_word_histogram_empty_tokens()
    print("all tests passed")
