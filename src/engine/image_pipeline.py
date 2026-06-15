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
from src.extraction.image_sift import extract_sift_features
from src.ml.clustering_trainer import KClustering
from src.ml.quantizer import VectorQuantizer


def _counts_to_hist(counts: np.ndarray, prefix: str) -> dict[str, int]:
    return {f"{prefix}_{i:04d}": int(c) for i, c in enumerate(counts) if c > 0}


def build_image_codebook(
    file_paths: Iterable[str | Path],
    codebook_size: int,
) -> list[np.ndarray]:
    all_descriptors = []
    for fp in file_paths:
        desc = extract_sift_features(str(fp))
        if desc.size > 0:
            all_descriptors.append(desc)
    if not all_descriptors:
        raise ValueError("Ninguna imagen produjo descriptores SIFT")
    big_matrix = np.vstack(all_descriptors)
    km = KClustering(n_centroids=codebook_size, clustering_algorithm="kmean")
    km.reset_centroids(dim=big_matrix.shape[1])
    km.clusterize(big_matrix)
    return km.close()


def _doc_stream(
    file_paths: Iterable[str | Path],
    quantizer: VectorQuantizer,
) -> Iterator[tuple[str, dict[str, int]]]:
    for fp in file_paths:
        desc = extract_sift_features(str(fp))
        if desc.size == 0:
            continue
        counts = quantizer.histogram(desc)
        hist = _counts_to_hist(counts, prefix="v")
        if hist:
            yield Path(fp).stem, hist


def index_image_corpus(
    file_paths: Iterable[str | Path],
    codebook_size: int,
    index_dir: str | Path,
    block_size_postings: int = 500_000,
) -> tuple[InvertedIndex, VectorQuantizer]:
    file_list = [Path(p) for p in file_paths]
    centroids = build_image_codebook(file_list, codebook_size)
    vq = VectorQuantizer(centroids)

    out = Path(index_dir)
    blocks = spimi_invert(
        _doc_stream(file_list, vq),
        block_size_postings=block_size_postings,
        out_dir=out / "blocks",
    )
    if not blocks:
        raise ValueError("Ningun documento produjo histograma; corpus vacio")
    postings_path, vocab_path = merge_blocks(blocks, out / "final")
    meta_path = build_meta(postings_path, vocab_path, out / "final")
    idx = InvertedIndex(postings_path, vocab_path, meta_path).open()
    return idx, vq


def prepare_image_query(
    image_path: str | Path,
    quantizer: VectorQuantizer,
) -> dict[str, int]:
    desc = extract_sift_features(str(image_path))
    if desc.size == 0:
        return {}
    counts = quantizer.histogram(desc)
    return _counts_to_hist(counts, prefix="v")


__all__ = [
    "build_image_codebook",
    "index_image_corpus",
    "prepare_image_query",
]