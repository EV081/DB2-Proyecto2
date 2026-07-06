from __future__ import annotations

from multiprocessing import cpu_count

import numpy as np


def counts_to_hist(counts: np.ndarray, prefix: str) -> dict[str, int]:
    return {f"{prefix}_{i:04d}": int(c) for i, c in enumerate(counts) if c > 0}


def pipeline_workers() -> int:
    return max(1, (cpu_count() or 1) - 1)


PARALLEL_THRESHOLD = 32