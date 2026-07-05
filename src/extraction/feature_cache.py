from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np

def feature_path(cache_dir: str | Path, stem: str) -> Path:
    return Path(cache_dir) / f"{stem}.npy"


def iter_cached(pairs: list[tuple[str, Path]]) -> Iterator[tuple[str, np.ndarray]]:
    for stem, p in pairs:
        yield stem, np.load(p, mmap_mode=None)


def aggregate_for_kmeans(
    pairs: list[tuple[str, Path]],
    max_samples: int | None,
    rng_seed: int = 42,
) -> np.ndarray:
    if not pairs:
        raise ValueError("aggregate_for_kmeans: pairs vacio")

    sizes = [int(np.load(p, mmap_mode="r").shape[0]) for _, p in pairs]
    total = sum(sizes)
    if total == 0:
        raise ValueError("aggregate_for_kmeans: 0 vectores totales")

    if max_samples and total > max_samples:
        rng = np.random.default_rng(rng_seed)
        sel = np.sort(rng.choice(total, size=max_samples, replace=False))
    else:
        sel = None

    chunks: list[np.ndarray] = []
    offset = 0
    for (_, p), n in zip(pairs, sizes):
        arr = np.load(p, mmap_mode=None)
        if sel is None:
            chunks.append(arr)
        else:
            local = sel[(sel >= offset) & (sel < offset + n)] - offset
            if len(local):
                chunks.append(arr[local])
        offset += n

    return np.vstack(chunks)


__all__ = ["feature_path", "iter_cached", "aggregate_for_kmeans"]
