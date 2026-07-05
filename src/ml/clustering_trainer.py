import numpy as np


class KClustering:

    def __init__(self, n_centroids=100, batch_size: int = 100_000, max_iter: int = 100, tol: float = 1e-6):
        self.centroids = np.empty((0, 0), dtype=np.float32)
        self.n_centroids = n_centroids
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol

    def reset_centroids(self, dim: int):
        self.centroids = np.random.randn(self.n_centroids, dim).astype(np.float32)

    def euclidean_distance(self, vector_a, vector_b):
        a = np.asarray(vector_a, dtype=np.float32)
        b = np.asarray(vector_b, dtype=np.float32)
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))

    def _assign_all(self, data: np.ndarray) -> np.ndarray:
        C = self.centroids
        c_sq = np.sum(C ** 2, axis=1)
        out = np.empty(len(data), dtype=np.int64)
        for start in range(0, len(data), self.batch_size):
            end = min(start + self.batch_size, len(data))
            Xb = data[start:end]
            x_sq = np.sum(Xb ** 2, axis=1, keepdims=True)
            dists_sq = x_sq + c_sq[None, :] - 2.0 * (Xb @ C.T)
            out[start:end] = dists_sq.argmin(axis=1)
        return out

    def assign_centroid(self, vector):
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return int(self._assign_all(v)[0])

    def _adjust_centroid_kmean(self, data, assignments):
        D = data.shape[1]
        new_centroids = np.zeros((self.n_centroids, D), dtype=np.float32)
        counts = np.bincount(assignments, minlength=self.n_centroids)
        np.add.at(new_centroids, assignments, data)
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty, None]
        # clusters vacios: conservar centroide anterior
        new_centroids[~nonempty] = self.centroids[~nonempty]
        self.centroids = new_centroids

    def clusterize(self, matrix):
        data = np.asarray(matrix, dtype=np.float32)
        n_points = len(data)
        self.n_centroids = min(self.n_centroids, n_points)
        if isinstance(self.centroids, list):
            self.centroids = np.asarray(self.centroids, dtype=np.float32)
        if len(self.centroids) > self.n_centroids:
            self.centroids = self.centroids[: self.n_centroids]

        assignments = np.zeros(n_points, dtype=np.int64)
        for _ in range(self.max_iter):
            assignments = self._assign_all(data)
            old = self.centroids.copy()
            self._adjust_centroid_kmean(data, assignments)
            movement = float(np.max(np.linalg.norm(self.centroids - old, axis=1)))
            if movement < self.tol:
                break

        return assignments, self.centroids

    def close(self):
        return [self.centroids[i] for i in range(len(self.centroids))]
