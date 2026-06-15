from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src.engine.inverted_index import (
    InvertedIndex,
    build_meta,
    merge_blocks,
    spimi_invert,
)
from src.extraction.audio_mfcc import extract_mfcc_features
from src.ml.clustering_trainer import KClustering
from src.ml.quantizer import VectorQuantizer


def _counts_to_hist(counts: np.ndarray, prefix: str) -> dict[str, int]:
    return {f"{prefix}_{i:04d}": int(c) for i, c in enumerate(counts) if c > 0}


def build_audio_codebook(
    file_paths: Iterable[str | Path],
    codebook_size: int,
    n_mfcc: int = 13,
) -> list[np.ndarray]:
    all_mfcc = []
    for fp in file_paths:
        mfcc = extract_mfcc_features(str(fp), n_mfcc=n_mfcc)
        if mfcc.size > 0:
            all_mfcc.append(mfcc)
    if not all_mfcc:
        raise ValueError("Ningun audio produjo coeficientes MFCC")
    big_matrix = np.vstack(all_mfcc)
    km = KClustering(n_centroids=codebook_size, clustering_algorithm="kmean")
    km.reset_centroids(dim=big_matrix.shape[1])
    km.clusterize(big_matrix)
    return km.close()


def _doc_stream(
    file_paths: Iterable[str | Path],
    quantizer: VectorQuantizer,
    n_mfcc: int,
) -> Iterator[tuple[str, dict[str, int]]]:
    for fp in file_paths:
        mfcc = extract_mfcc_features(str(fp), n_mfcc=n_mfcc)
        if mfcc.size == 0:
            continue
        counts = quantizer.histogram(mfcc)
        hist = _counts_to_hist(counts, prefix="a")
        if hist:
            yield Path(fp).stem, hist


def index_audio_corpus(
    file_paths: Iterable[str | Path],
    codebook_size: int,
    index_dir: str | Path,
    n_mfcc: int = 13,
    block_size_postings: int = 500_000,
) -> tuple[InvertedIndex, VectorQuantizer]:
    file_list = [Path(p) for p in file_paths]
    centroids = build_audio_codebook(file_list, codebook_size, n_mfcc=n_mfcc)
    vq = VectorQuantizer(centroids)

    out = Path(index_dir)
    blocks = spimi_invert(
        _doc_stream(file_list, vq, n_mfcc),
        block_size_postings=block_size_postings,
        out_dir=out / "blocks",
    )
    if not blocks:
        raise ValueError("Ningun documento produjo histograma; corpus vacio")
    postings_path, vocab_path = merge_blocks(blocks, out / "final")
    meta_path = build_meta(postings_path, vocab_path, out / "final")
    idx = InvertedIndex(postings_path, vocab_path, meta_path).open()
    return idx, vq


def prepare_audio_query(
    audio_path: str | Path,
    quantizer: VectorQuantizer,
    n_mfcc: int = 13,
) -> dict[str, int]:
    mfcc = extract_mfcc_features(str(audio_path), n_mfcc=n_mfcc)
    if mfcc.size == 0:
        return {}
    counts = quantizer.histogram(mfcc)
    return _counts_to_hist(counts, prefix="a")


__all__ = [
    "build_audio_codebook",
    "index_audio_corpus",
    "prepare_audio_query",
]