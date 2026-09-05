"""K-Means candidate fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass(frozen=True, slots=True)
class KCandidate:
    k: int
    silhouette: float
    inertia: float
    calinski_harabasz: float | None
    davies_bouldin: float | None
    min_cluster_size: int
    max_cluster_size: int
    labels: tuple[int, ...]


def search_kmeans(matrix: np.ndarray, *, max_k: int = 8, random_state: int = 42) -> tuple[KCandidate, ...]:
    n_rows = matrix.shape[0]
    upper = min(max_k, n_rows - 1)
    results: list[KCandidate] = []
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(matrix)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or len(unique) >= n_rows:
            continue
        try:
            sil = float(silhouette_score(matrix, labels))
        except ValueError:
            continue
        ch = float(calinski_harabasz_score(matrix, labels)) if n_rows > len(unique) else None
        db = float(davies_bouldin_score(matrix, labels)) if len(unique) > 1 else None
        results.append(KCandidate(
            k=k,
            silhouette=sil,
            inertia=float(model.inertia_),
            calinski_harabasz=ch,
            davies_bouldin=db,
            min_cluster_size=int(counts.min()),
            max_cluster_size=int(counts.max()),
            labels=tuple(int(x) for x in labels),
        ))
    return tuple(results)
