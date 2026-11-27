
import numpy as np


def calculate_reachability_distances(
    distance_matrix: np.ndarray,
    minpts_distances: np.ndarray
) -> np.ndarray:
    """
    Compute directed reachability distances for all observation pairs.

    reachdist(p, o) = max(k-distance(o), distance(p, o))

    Args:
        distance_matrix: Pairwise Euclidean distances, shape (n, n)
        minpts_distances: MinPts-distance for each observation, shape (n,)

    Returns:
        Reachability distance matrix of shape (n, n)
        where entry [p, o] corresponds to reachdist(p, o)
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    k_dist = np.asarray(minpts_distances, dtype=np.float64)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")

    n = D.shape[0]

    if n < 2:
        raise ValueError("At least two observations are required.")

    if k_dist.ndim != 1 or k_dist.shape[0] != n:
        raise ValueError(
            "MinPts-distances must be a 1D array of length n."
        )

    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(k_dist)):
        raise ValueError("Distances must be finite.")

    if np.any(D < 0) or np.any(k_dist < 0):
        raise ValueError("Distances cannot be negative.")

    if not np.allclose(D, D.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Euclidean distance matrix must be symmetric.")

    # Apply the MinPts-distance of the neighbor o (columns).
    return np.maximum(D, k_dist[np.newaxis, :])
