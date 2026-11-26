import numpy as np

from ._utils import NeighborhoodInput, is_rectangular_neighborhoods, pad_neighborhoods


def calculate_lrd_loops(
    neighborhood_indices: NeighborhoodInput,
    reachability_distances: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Calculates the Local Reachability Density (LRD) for each point using loops.

    Args:
        neighborhood_indices: An (N, k) array or length-N sequence of 1D neighbor index arrays.
        reachability_distances: An (N, N) distance matrix where entry [p, o] is
                                the reachability distance between p and o.
        epsilon: A small constant added to the mean reachability distance to prevent division by zero.

    Returns:
        An array of size N containing the LRD value for each point.
    """
    n_points = len(neighborhood_indices)
    if n_points == 0:
        return np.empty(0, dtype=float)

    lrd = np.zeros(n_points, dtype=float)

    for p in range(n_points):
        neighbors = neighborhood_indices[p]
        k_p = len(neighbors)
        if k_p == 0:
            raise ValueError("Every point must have at least one neighbor.")

        sum_reach_dist = 0.0

        # 1. Sum reachability distances to all neighbors in N(p)
        for o in neighbors:
            sum_reach_dist += reachability_distances[p, o]

        # 2 & 3. Stabilize mean reachability distance with epsilon and invert for LRD
        mean_reach_dist = sum_reach_dist / k_p
        lrd[p] = 1.0 / (mean_reach_dist + epsilon)

    # 4. Return array of LRD values
    return lrd


def calculate_lrd_vectorized(
    neighborhood_indices: NeighborhoodInput,
    reachability_distances: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Calculates the Local Reachability Density (LRD) for each point using vectorized NumPy operations.

    Args:
        neighborhood_indices: An (N, k) array or length-N sequence of 1D neighbor index arrays.
        reachability_distances: An (N, N) distance matrix where entry [p, o] is
                                the reachability distance between p and o.
        epsilon: A small constant added to the mean reachability distance to prevent division by zero.

    Returns:
        An array of size N containing the LRD value for each point.
    """
    if is_rectangular_neighborhoods(neighborhood_indices):
        n_points, k_neighbors = neighborhood_indices.shape
        if n_points == 0:
            return np.empty(0, dtype=float)
        if k_neighbors == 0:
            raise ValueError("Every point must have at least one neighbor.")

        row_indices = np.arange(n_points)[:, np.newaxis]
        neighborhood_reach_dists = reachability_distances[row_indices, neighborhood_indices]
        mean_reach_dist = np.mean(neighborhood_reach_dists, axis=1)
        return 1.0 / (mean_reach_dist + epsilon)

    padded_indices, mask, counts = pad_neighborhoods(neighborhood_indices)
    n_points = len(counts)
    if n_points == 0:
        return np.empty(0, dtype=float)

    row_indices = np.arange(n_points)[:, np.newaxis]
    neighborhood_reach_dists = reachability_distances[row_indices, padded_indices]
    masked_reach_dists = np.where(mask, neighborhood_reach_dists, 0.0)
    mean_reach_dist = np.sum(masked_reach_dists, axis=1) / counts
    return 1.0 / (mean_reach_dist + epsilon)
