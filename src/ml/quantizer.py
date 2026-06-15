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

    def __init__(self, n_centroids=100, clustering_algorithm="kmean", n_init=5):
        """
        k_centroids: número de centroides o listas (termino de pgvector)
        clustering_algorithm: algoritmo usado para la clusterización
        n_init: veces que se repetirá el algoritmo
        """
        self.centroids = []
        self.n_centroids = n_centroids
        self.clustering_algorithm = clustering_algorithm
        self.n_init = n_init

    def euclidean_distance(self, vector_a, vector_b):
        a = np.asarray(vector_a)
        b = np.asarray(vector_b)
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))

    def assign_centroid(self, vector):
        """
        Devuelve el índice del centroide más cercano al vector dado.
        """
        best_dist = float('inf')
        best_idx = 0
        for i, centroid in enumerate(self.centroids):
            dist = self.euclidean_distance(vector, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _adjust_centroid_kmean(self, data, assignments):
        """
        Cada centroide = promedio de los vectores asignados a él.
        """
        for i in range(self.n_centroids):
            members = []
            for j in range(len(data)):
                if assignments[j] == i:
                    members.append(data[j])
            if members:
                self.centroids[i] = np.mean(members, axis=0)

    def _adjust_centroid_kmedoid(self, data, assignments):
        """
        Cada centroide = punto del cluster que minimiza la suma
        de distancias a los demás miembros.
        """
        for i in range(self.n_centroids):
            members_idx = []
            for j in range(len(data)):
                if assignments[j] == i:
                    members_idx.append(j)
            if members_idx:
                best_idx = members_idx[0]
                best_cost = float('inf')
                for candidate in members_idx:
                    cost = 0.0
                    for m in members_idx:
                        cost += self.euclidean_distance(
                            data[candidate], data[m]
                        )
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = candidate
                self.centroids[i] = data[best_idx].copy()

    def _acumulate_error(self, data, assignments):
        """
        Suma de distancias de cada punto al centroide que le tocó.
        Mide qué tan compactos son los clusters.
        """
        total = 0.0
        for i in range(len(data)):
            total += self.euclidean_distance(
                data[i], self.centroids[assignments[i]]
            )
        return total

    def nearest_centroid(self, vector):
        """
        Para un vector nuevo, devuelve (índice, centroide) del más cercano.
        """
        best_dist = float('inf')
        best_idx = 0
        for i, centroid in enumerate(self.centroids):
            dist = self.euclidean_distance(vector, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, self.centroids[best_idx]

    def clusterize_by_matrix(self, matrix):
        data = np.asarray(matrix)
        n_points = len(data)
        self.n_centroids = min(self.n_centroids, n_points)

        best_centroids = None
        best_assignments = None
        best_error = float('inf')

        for iter_1 in range(self.n_init):
            random_idx = np.random.choice(n_points, self.n_centroids, replace=False)
            self.centroids = [data[i].copy() for i in random_idx]

            for iter_2 in range(100):
                assignments = [-1] * n_points
                for i in range(n_points):
                    assignments[i] = self.assign_centroid(data[i])

                old_centroids = [c.copy() for c in self.centroids]

                if self.clustering_algorithm == "kmean":
                    self._adjust_centroid_kmean(data, assignments)
                elif self.clustering_algorithm == "kmedoid":
                    self._adjust_centroid_kmedoid(data, assignments)
                else:
                    raise ValueError(
                        f"Algoritmo desconocido: '{self.clustering_algorithm}'. "
                        "Usa 'kmean' o 'kmedoid'."
                    )

                max_movement = 0.0
                for old, new in zip(old_centroids, self.centroids):
                    movement = self.euclidean_distance(old, new)
                    if movement > max_movement:
                        max_movement = movement

                if max_movement < 1e-6:
                    break

            error = self._acumulate_error(data, assignments)
            if error < best_error:
                best_error = error
                best_centroids = [c.copy() for c in self.centroids]
                best_assignments = assignments[:]

        self.centroids = best_centroids
        return np.array(best_assignments), best_centroids



if __name__ == "__main__":

    data = np.array([1, 2, 4, 5, 9, 10, 14, 18, 22], dtype=float).reshape(-1, 1)

    km = KClustering(n_centroids=3, clustering_algorithm="kmean")
    labels, centroids = km.clusterize_by_matrix(data)

    print("Datos originales:", data.ravel())
    print("Centroides finales:", np.array(centroids).ravel())
    print()

    for i in range(km.n_centroids):
        cluster_points = data[labels == i].ravel()
        print(f"Cluster {i}: {cluster_points}  (centroide={np.array(centroids[i]).ravel()})")

    nuevo_punto = np.array([[7.0]])
    idx, vec = km.nearest_centroid(nuevo_punto)
    print(f"\nPunto nuevo {nuevo_punto.ravel()} -> cluster {idx} (centroide={vec})")
