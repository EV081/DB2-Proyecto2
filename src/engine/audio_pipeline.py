from __future__ import annotations
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np
from src.engine._utils import PARALLEL_THRESHOLD, counts_to_hist, pipeline_workers
from src.engine.inverted_index import (
    InvertedIndex,
    build_meta,
    merge_blocks,
    spimi_invert,
)
from src.extraction.audio_mfcc import extract_mfcc_features
from src.extraction.feature_cache import (
    aggregate_for_kmeans,
    feature_path,
    iter_cached,
)
from src.ml.clustering_trainer import KClustering
from src.ml.quantizer import VectorQuantizer


def _extract_mfcc_worker(args):
    fp_str, n_mfcc = args
    try:
        mfcc = extract_mfcc_features(fp_str, n_mfcc=n_mfcc)
        return fp_str, mfcc, None
    except Exception as e:
        return fp_str, None, f"{e.__class__.__name__}: {e}"


def _cache_mfcc_worker(args):
    fp_str, cache_path_str, n_mfcc = args
    cache_path = Path(cache_path_str)
    try:
        if cache_path.exists():
            return fp_str, str(cache_path), None
        mfcc = extract_mfcc_features(fp_str, n_mfcc=n_mfcc)
        if mfcc.size == 0:
            return fp_str, None, "empty"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mfcc)
        return fp_str, str(cache_path), None
    except Exception as e:
        return fp_str, None, f"{e.__class__.__name__}: {e}"


def cache_audio_features(
    file_paths: Iterable[str | Path],
    cache_dir: str | Path,
    n_mfcc: int = 13,
    log_label: str = "audio cache",
) -> list[tuple[str, Path]]:
    files = [Path(fp) for fp in file_paths]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    args = [(str(fp), str(feature_path(cache_dir, fp.stem)), n_mfcc) for fp in files]
    n_total = len(args)
    n_workers = pipeline_workers()
    pairs: list[tuple[str, Path]] = []
    n_skipped = 0
    n_done = 0
    log_every = max(50, n_total // 20)

    if n_total >= PARALLEL_THRESHOLD and n_workers > 1:
        print(f"  [{log_label}] {n_total} archivos en {n_workers} workers paralelos")
        with Pool(n_workers) as pool:
            for fp_str, cp, err in pool.imap_unordered(
                _cache_mfcc_worker, args, chunksize=10,
            ):
                n_done += 1
                if err is not None:
                    n_skipped += 1
                    print(f"  [SKIP {log_label}] {Path(fp_str).name}: {err}")
                else:
                    pairs.append((Path(fp_str).stem, Path(cp)))
                if n_done % log_every == 0:
                    print(f"  [{log_label}] {n_done}/{n_total}")
    else:
        for arg in args:
            fp_str, cp, err = _cache_mfcc_worker(arg)
            n_done += 1
            if err is not None:
                n_skipped += 1
                print(f"  [SKIP {log_label}] {Path(fp_str).name}: {err}")
            else:
                pairs.append((Path(fp_str).stem, Path(cp)))

    if n_skipped:
        print(f"  [{log_label}] {n_skipped} archivos saltados")
    return pairs


def _train_codebook_from_pairs(
    pairs: list[tuple[str, Path]],
    codebook_size: int,
    max_samples: int,
    rng_seed: int = 42,
) -> list[np.ndarray]:
    if not pairs:
        raise ValueError("Ningun audio produjo coeficientes MFCC")
    big_matrix = aggregate_for_kmeans(pairs, max_samples=max_samples, rng_seed=rng_seed)
    n_vec = big_matrix.shape[0]
    if max_samples and n_vec >= max_samples:
        print(f"  [subsample] KMeans sobre {n_vec} vectores MFCC")
    km = KClustering(n_centroids=codebook_size)
    km.reset_centroids(dim=big_matrix.shape[1])
    km.clusterize(big_matrix)
    return km.close()


def build_audio_codebook(
    file_paths: Iterable[str | Path],
    codebook_size: int,
    n_mfcc: int = 13,
    max_samples: int = 50_000,
    rng_seed: int = 42,
    cache_dir: str | Path | None = None,
) -> list[np.ndarray]:
    file_list = [Path(p) for p in file_paths]
    if cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="audio_features_") as td:
            pairs = cache_audio_features(file_list, td, n_mfcc=n_mfcc)
            return _train_codebook_from_pairs(pairs, codebook_size, max_samples, rng_seed)
    pairs = cache_audio_features(file_list, cache_dir, n_mfcc=n_mfcc)
    return _train_codebook_from_pairs(pairs, codebook_size, max_samples, rng_seed)


def _doc_stream_from_cache(
    pairs: list[tuple[str, Path]],
    quantizer: VectorQuantizer,
) -> Iterator[tuple[str, dict[str, int]]]:
    n_total = len(pairs)
    log_every = max(200, n_total // 20)
    for i, (stem, arr) in enumerate(iter_cached(pairs), 1):
        counts = quantizer.histogram(arr)
        hist = counts_to_hist(counts, prefix="a")
        if hist:
            yield stem, hist
        if i % log_every == 0:
            print(f"  [audio quant] {i}/{n_total}")


def index_audio_corpus(
    file_paths: Iterable[str | Path],
    codebook_size: int,
    index_dir: str | Path,
    n_mfcc: int = 13,
    block_size_postings: int = 500_000,
    max_samples: int = 50_000,
) -> tuple[InvertedIndex, VectorQuantizer]:
    file_list = [Path(p) for p in file_paths]
    out = Path(index_dir)
    cache = out / "_features"

    pairs = cache_audio_features(file_list, cache, n_mfcc=n_mfcc)
    if not pairs:
        raise ValueError("Ningun audio produjo features cacheadas")

    centroids = _train_codebook_from_pairs(pairs, codebook_size, max_samples)
    vq = VectorQuantizer(centroids)

    blocks = spimi_invert(
        _doc_stream_from_cache(pairs, vq),
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
    return counts_to_hist(counts, prefix="a")


__all__ = [
    "build_audio_codebook",
    "cache_audio_features",
    "index_audio_corpus",
    "prepare_audio_query",
]
