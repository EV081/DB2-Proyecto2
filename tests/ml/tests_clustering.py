from __future__ import annotations

import numpy as np

from src.ml.clustering_trainer import KClustering
from src.ml.quantizer import VectorQuantizer

# Dataset 1D con 3 grupos naturales: {1,2,4,5}, {9,10,14}, {18,22}
_DATA_1D = np.array([1, 2, 4, 5, 9, 10, 14, 18, 22], dtype=float).reshape(-1, 1)


def _run_clustering() -> tuple[np.ndarray, list[np.ndarray]]:
    np.random.seed(42)
    km = KClustering(n_centroids=3)
    km.reset_centroids(dim=1)
    labels, _ = km.clusterize(_DATA_1D)
    return labels, km.close()


def test_kmeans_groups_close_points_together() -> None:
    labels, _ = _run_clustering()
    # los dos puntos mas pequenos comparten cluster
    assert labels[0] == labels[1]
    # los dos puntos mas grandes comparten cluster
    assert labels[7] == labels[8]
    # los extremos NO comparten cluster
    assert labels[0] != labels[8]
    # hay exactamente 3 clusters distintos
    assert len(set(labels.tolist())) == 3


def test_kmeans_produces_3_centroids_of_expected_dim() -> None:
    _, centroids = _run_clustering()
    assert len(centroids) == 3
    for c in centroids:
        assert c.shape == (1,)


def test_vector_quantizer_maps_point_to_nearest_cluster() -> None:
    _, centroids = _run_clustering()
    vq = VectorQuantizer(centroids)
    idx, vec = vq.nearest_centroid(np.array([[7.0]]))
    assert 0 <= idx < 3
    # el centroide devuelto es el mas cercano a 7 entre los 3
    best_dist = abs(float(vec[0]) - 7.0)
    for c in centroids:
        assert abs(float(c[0]) - 7.0) >= best_dist - 1e-6


if __name__ == "__main__":
    test_kmeans_groups_close_points_together()
    test_kmeans_produces_3_centroids_of_expected_dim()
    test_vector_quantizer_maps_point_to_nearest_cluster()
    print("all tests passed")
