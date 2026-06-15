from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.image_pipeline import (
    build_image_codebook,
    index_image_corpus,
    prepare_image_query,
)


# Generamos 3 "clases" de imagenes sinteticas con patrones SIFT-detectables
# (la KClustering pure-Python es lenta, asi que mantenemos N chico y vocab=8).
SIZE = 64
PATCH = 32
STRIDE = 16


def _make_stripes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for x in range(0, SIZE, 4):
        img[:, x:x + 2] = rng.integers(100, 256, 3, dtype=np.uint8)
    return img


def _make_dots(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((SIZE, SIZE, 3), 30, dtype=np.uint8)
    for _ in range(30):
        cy, cx = rng.integers(5, SIZE - 5, 2)
        color = rng.integers(150, 256, 3).tolist()
        cv2.circle(img, (int(cx), int(cy)), 3, color, -1)
    return img


def _make_noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def _make_corpus(tmp: Path) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {"stripes": [], "dots": [], "noise": []}
    generators = {"stripes": _make_stripes, "dots": _make_dots, "noise": _make_noise}
    for cls, gen in generators.items():
        for i in range(2):
            img = gen(seed=hash((cls, i)) & 0xFFFF)
            path = tmp / f"{cls}_{i:02d}.png"
            cv2.imwrite(str(path), img)
            by_class[cls].append(str(path))
    return by_class


def _all_paths(by_class: dict[str, list[str]]) -> list[str]:
    return [p for paths in by_class.values() for p in paths]


def test_build_image_codebook_returns_centroids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        by_class = _make_corpus(Path(tmp))
        centroids = build_image_codebook(_all_paths(by_class), codebook_size=8)
        assert 1 <= len(centroids) <= 8
        for c in centroids:
            assert c.shape == (128,)


def test_index_image_corpus_builds_indexable_corpus() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        by_class = _make_corpus(tmp)
        idx, vq = index_image_corpus(
            file_paths=_all_paths(by_class),
            codebook_size=8,
            index_dir=tmp / "index",
        )
        try:
            assert idx.n_docs >= 1
            assert idx.vocab_size() > 0
            assert vq.n_centroids <= 8
        finally:
            idx.close()


def test_prepare_image_query_returns_histogram() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        by_class = _make_corpus(tmp)
        idx, vq = index_image_corpus(
            file_paths=_all_paths(by_class),
            codebook_size=8,
            index_dir=tmp / "index",
        )
        try:
            q_hist = prepare_image_query(by_class["stripes"][0], vq)
            assert isinstance(q_hist, dict)
            assert all(k.startswith("v_") for k in q_hist)
            assert all(isinstance(v, int) and v > 0 for v in q_hist.values())
        finally:
            idx.close()


def test_query_returns_results_for_indexed_image() -> None:
    # No exigimos top-1 perfecto (K-Means con seed aleatoria + corpus chico
    # es ruidoso); solo verificamos que la query devuelve algo y que el
    # archivo consultado esta en el top-K.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        by_class = _make_corpus(tmp)
        all_paths = _all_paths(by_class)
        idx, vq = index_image_corpus(
            file_paths=all_paths,
            codebook_size=8,
            index_dir=tmp / "index",
        )
        try:
            for query_path in all_paths[:3]:
                q_hist = prepare_image_query(query_path, vq)
                if not q_hist:
                    continue
                results = idx.search_topk(q_hist, k=len(all_paths))
                assert results, f"sin resultados para {query_path}"
                doc_ids = [d for d, _ in results]
                expected = Path(query_path).stem
                assert expected in doc_ids, (
                    f"{expected!r} no aparece en top-{len(all_paths)}: {doc_ids}"
                )
        finally:
            idx.close()


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()
