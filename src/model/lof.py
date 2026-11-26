import numpy as np

from ._utils import NeighborhoodInput, is_rectangular_neighborhoods, pad_neighborhoods


def calculate_lof_iterative(
    neighborhood_indices: NeighborhoodInput,
    lrd_array: np.ndarray
) -> np.ndarray:
    """
    Calculate Local Outlier Factor (LOF) for each point using explicit loops.

    Args:
        neighborhood_indices: An (N, k) array or length-N sequence of 1D neighbor index arrays.
        lrd_array (np.ndarray): A 1D array of length N containing the Local Reachability Density.

    Returns:
        np.ndarray: A 1D array of length N containing the LOF scores.
    """
    n_points = len(neighborhood_indices)
    if n_points == 0:
        return np.empty(0, dtype=float)

    lof_scores = np.zeros(n_points, dtype=float)

    # 1. For each point p...
    for p in range(n_points):
        neighbors = neighborhood_indices[p]
        k_p = len(neighbors)
        if k_p == 0:
            raise ValueError("Every point must have at least one neighbor.")

        lrd_p = lrd_array[p]
        sum_ratios = 0.0

        # ...iterate through its neighbors (o in N_MinPts[p])
        for o in neighbors:
            lrd_o = lrd_array[o]

            # 2. Calculate the ratio of the neighbor's LRD to the point's own LRD
            # 3. Sum all these ratios
            sum_ratios += lrd_o / lrd_p

        # 4. Divide the sum by the number of neighbors
        lof_scores[p] = sum_ratios / k_p

    # 5. Return array of LOF scores
    return lof_scores


def calculate_lof_vectorized(
    neighborhood_indices: NeighborhoodInput,
    lrd_array: np.ndarray
) -> np.ndarray:
    """
    Calculate Local Outlier Factor (LOF) using efficient vectorized NumPy operations.

    Args:
        neighborhood_indices: An (N, k) array or length-N sequence of 1D neighbor index arrays.
        lrd_array (np.ndarray): A 1D array of length N containing the Local Reachability Density.

    Returns:
        np.ndarray: A 1D array of length N containing the LOF scores.
    """
    if is_rectangular_neighborhoods(neighborhood_indices):
        n_points, k = neighborhood_indices.shape
        if n_points == 0:
            return np.empty(0, dtype=float)
        if k == 0:
            raise ValueError("Every point must have at least one neighbor.")

        lrd_neighbors = lrd_array[neighborhood_indices]
        return np.mean(lrd_neighbors / lrd_array[:, np.newaxis], axis=1)

    padded_indices, mask, counts = pad_neighborhoods(neighborhood_indices)
    if len(counts) == 0:
        return np.empty(0, dtype=float)

    lrd_neighbors = np.where(mask, lrd_array[padded_indices], 0.0)
    sum_lrd_neighbors = np.sum(lrd_neighbors, axis=1)
    return sum_lrd_neighbors / (counts * lrd_array)
