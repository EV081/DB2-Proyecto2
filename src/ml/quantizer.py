"""
==== Módulo KClustering ====

Recibe una matriz (array de vectores) de puntos y aplica
algún algoritmo de clustering:
 - "kmean"
 - "kmedoid"

Se pueden definir la cantidad de centroides y el algoritmo.

"""

import numpy as np


class KClustering:

    def __init__(self, k_centroids=100, clustering_algorithm="kmean", n_init=10):
        self.centroids = []
        self.k_centroids = k_centroids
        self.clustering_algorithm = clustering_algorithm
        self.n_init = n_init

    def euclidean_distance(self, vector_a, vector_b):
        return float(np.linalg.norm(np.asarray(vector_a) - np.asarray(vector_b)))

    def nearest_centroid(self, vector):
        vector = np.asarray(vector)
        centroids_arr = np.asarray(self.centroids)
        distances = np.linalg.norm(centroids_arr - vector, axis=1)
        nearest_index = int(np.argmin(distances))
        return nearest_index, self.centroids[nearest_index]

    def _kmeans_run(self, points):
        n_samples = points.shape[0]
        indices = np.random.choice(n_samples, self.k_centroids, replace=False)
        centroids = points[indices].tolist()

        for _ in range(100):
            centroids_arr = np.asarray(centroids)
            pairwise = np.linalg.norm(points[:, np.newaxis] - centroids_arr, axis=2)
            labels = np.argmin(pairwise, axis=1)

            new_centroids = []
            for i in range(self.k_centroids):
                mask = labels == i
                if np.any(mask):
                    new_centroids.append(points[mask].mean(axis=0))
                else:
                    new_centroids.append(centroids[i])

            new_centroids_arr = np.asarray(new_centroids)
            shift = np.linalg.norm(new_centroids_arr - centroids_arr)
            centroids = new_centroids_arr.tolist()

            if shift < 1e-6:
                break

        centroids_arr = np.asarray(centroids)
        pairwise = np.linalg.norm(points[:, np.newaxis] - centroids_arr, axis=2)
        inertia = float(np.sum(np.min(pairwise ** 2, axis=1)))
        return labels, centroids, inertia

    def _kmeans(self, points):
        best_labels = None
        best_centroids = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            labels, centroids, inertia = self._kmeans_run(points)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.centroids = best_centroids
        return best_labels, self.centroids

    def _kmedoids_run(self, points):
        n_samples = points.shape[0]
        indices = np.random.choice(n_samples, self.k_centroids, replace=False)
        centroids = points[indices].tolist()

        for _ in range(100):
            centroids_arr = np.asarray(centroids)
            pairwise = np.linalg.norm(points[:, np.newaxis] - centroids_arr, axis=2)
            labels = np.argmin(pairwise, axis=1)

            new_centroids = []
            for i in range(self.k_centroids):
                mask = labels == i
                if np.any(mask):
                    cluster_points = points[mask]
                    cluster_dists = np.linalg.norm(
                        cluster_points[:, np.newaxis] - cluster_points, axis=2
                    )
                    medoid_idx = int(np.argmin(cluster_dists.sum(axis=1)))
                    new_centroids.append(cluster_points[medoid_idx])
                else:
                    new_centroids.append(centroids[i])

            new_centroids_arr = np.asarray(new_centroids)
            shift = np.linalg.norm(new_centroids_arr - centroids_arr)
            centroids = new_centroids_arr.tolist()

            if shift < 1e-6:
                break

        centroids_arr = np.asarray(centroids)
        pairwise = np.linalg.norm(points[:, np.newaxis] - centroids_arr, axis=2)
        inertia = float(np.sum(np.min(pairwise ** 2, axis=1)))
        return labels, centroids, inertia

    def _kmedoids(self, points):
        best_labels = None
        best_centroids = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            labels, centroids, inertia = self._kmedoids_run(points)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.centroids = best_centroids
        return best_labels, self.centroids

    def clusterize_by_matrix(self, matrix):
        points = np.asarray(matrix)

        if self.clustering_algorithm == "kmean":
            return self._kmeans(points)
        elif self.clustering_algorithm == "kmedoid":
            return self._kmedoids(points)
        else:
            raise ValueError(
                f"Unknown algorithm: {self.clustering_algorithm}. "
                "Use 'kmean' or 'kmedoid'."
            )


if __name__ == "__main__":
    import numpy as np

    data = np.array([1, 2, 4, 5, 9, 10, 14, 18, 22], dtype=float).reshape(-1, 1)

    km = KClustering(k_centroids=3, clustering_algorithm="kmean")
    labels, centroids = km.clusterize_by_matrix(data)

    print("Datos originales:", data.ravel())
    print("Centroides finales:", np.array(centroids).ravel())
    print()

    for i in range(km.k_centroids):
        cluster_points = data[labels == i].ravel()
        print(f"Cluster {i}: {cluster_points}  (centroide={np.array(centroids[i]).ravel()})")

    nuevo_punto = np.array([[7.0]])
    idx, vec = km.nearest_centroid(nuevo_punto)
    print(f"\nPunto nuevo {nuevo_punto.ravel()} -> cluster {idx} (centroide={vec})")
