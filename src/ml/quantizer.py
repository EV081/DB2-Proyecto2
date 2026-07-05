import numpy as np


class VectorQuantizer:

    def __init__(self, centroids, batch_size: int = 100_000):
        arr = np.asarray(centroids, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.centroids = arr
        self.n_centroids = len(arr)
        self.batch_size = batch_size
        self._c_sq = np.sum(arr ** 2, axis=1)

    def _assign_batch(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), dtype=np.int64)
        C = self.centroids
        c_sq = self._c_sq
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            Xb = X[start:end]
            x_sq = np.sum(Xb ** 2, axis=1, keepdims=True)
            dists_sq = x_sq + c_sq[None, :] - 2.0 * (Xb @ C.T)
            out[start:end] = dists_sq.argmin(axis=1)
        return out

    def nearest_centroid(self, vector):
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        idx = int(self._assign_batch(v)[0])
        return idx, self.centroids[idx]

    def histogram(self, matrix):
        X = np.asarray(matrix, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assignments = self._assign_batch(X)
        counts = np.bincount(assignments, minlength=self.n_centroids)
        return counts.astype(int)


class WordQuantizer:

    def __init__(self, bag_of_words):
        self.bag_of_words = list(bag_of_words)
        self.n_words = len(bag_of_words)
        self._index = {w: i for i, w in enumerate(self.bag_of_words)}

    def histogram(self, tokens):
        counts = np.zeros(self.n_words, dtype=int)
        for token in tokens:
            i = self._index.get(token)
            if i is not None:
                counts[i] += 1
        return counts
