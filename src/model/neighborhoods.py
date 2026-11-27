
import numpy as np


def calculate_minpts_neighborhoods(
    distance_matrix: np.ndarray,
    min_pts: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Calculate MinPts-distances and tie-inclusive neighborhoods.

    Args:
        distance_matrix: Square pairwise Euclidean distance matrix.
        min_pts: Number of nearest neighbors defining the boundary,
                 excluding the observation itself.

    Returns:
        minpts_distances: Shape (n_samples,).
        neighborhood_indices: List of neighbor-index arrays.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)

    # Validate the distance matrix.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")

    n = D.shape[0]

    if (
        isinstance(min_pts, (bool, np.bool_))
        or not isinstance(min_pts, (int, np.integer))
        or not 1 <= min_pts < n
    ):
        raise ValueError("min_pts must be between 1 and n_samples - 1.")

    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative.")

    if not np.array_equal(np.diag(D), np.zeros(n)):
        raise ValueError("Distance matrix diagonal must be zero.")

    if not np.allclose(D, D.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric.")

    # The diagonal contains each observation's distance to itself (0).
    # Therefore index min_pts gives the min_pts-th OTHER distance.
    minpts_distances = np.partition(
        D, kth=min_pts, axis=1
    )[:, min_pts]

    neighborhood_indices = []

    for p in range(n):
        # Include all points at or inside the MinPts-distance.
        neighbors = np.flatnonzero(D[p] <= minpts_distances[p])

        # Exclude the observation itself, including for duplicate points.
        neighbors = neighbors[neighbors != p]

        neighborhood_indices.append(neighbors)

    return minpts_distances, neighborhood_indices
