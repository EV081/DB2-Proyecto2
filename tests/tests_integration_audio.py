from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.engine.audio_pipeline import (
    build_audio_codebook,
    index_audio_corpus,
    prepare_audio_query,
)


# 3 "clases" de audio: tonos senoidales a frecuencias distintas.
# Duracion corta para que la KClustering pure-Python termine rapido.
SR = 22050
DURATION_S = 1.0


def _make_tone(freq_hz: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    signal += 0.02 * rng.standard_normal(signal.shape)  # ruido leve
    return signal.astype(np.float32)


def _make_corpus(tmp: Path) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {"low": [], "mid": [], "high": []}
    freqs = {"low": 220.0, "mid": 440.0, "high": 880.0}
    for cls, freq in freqs.items():
        for i in range(2):
            sig = _make_tone(freq, seed=hash((cls, i)) & 0xFFFF)
            path = tmp / f"{cls}_{i:02d}.wav"
            sf.write(str(path), sig, SR)
            by_class[cls].append(str(path))
    return by_class


def _all_paths(by_class: dict[str, list[str]]) -> list[str]:
    return [p for paths in by_class.values() for p in paths]


def test_build_audio_codebook_returns_centroids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        by_class = _make_corpus(Path(tmp))
        centroids = build_audio_codebook(_all_paths(by_class), codebook_size=6)
        assert 1 <= len(centroids) <= 6
        for c in centroids:
            assert c.shape == (13,)


def test_index_audio_corpus_builds_indexable_corpus() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        by_class = _make_corpus(tmp)
        idx, vq = index_audio_corpus(
            file_paths=_all_paths(by_class),
            codebook_size=6,
            index_dir=tmp / "index",
        )
        try:
            assert idx.n_docs == 6   # 3 clases x 2 archivos
            assert idx.vocab_size() > 0
            assert vq.n_centroids <= 6
        finally:
            idx.close()


def test_prepare_audio_query_returns_histogram() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        by_class = _make_corpus(tmp)
        idx, vq = index_audio_corpus(
            file_paths=_all_paths(by_class),
            codebook_size=6,
            index_dir=tmp / "index",
        )
        try:
            q_hist = prepare_audio_query(by_class["low"][0], vq)
            assert isinstance(q_hist, dict)
            assert q_hist, "histograma vacio en query"
            assert all(k.startswith("a_") for k in q_hist)
            assert all(isinstance(v, int) and v > 0 for v in q_hist.values())
        finally:
            idx.close()


def test_query_returns_same_file_in_top_results() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        by_class = _make_corpus(tmp)
        all_paths = _all_paths(by_class)
        idx, vq = index_audio_corpus(
            file_paths=all_paths,
            codebook_size=6,
            index_dir=tmp / "index",
        )
        try:
            for query_path in all_paths[:3]:
                q_hist = prepare_audio_query(query_path, vq)
                assert q_hist, f"query vacia para {query_path}"
                results = idx.search_topk(q_hist, k=len(all_paths))
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
