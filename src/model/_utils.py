from typing import Sequence, Union
import numpy as np

NeighborhoodInput = Union[np.ndarray, Sequence[np.ndarray]]


def is_rectangular_neighborhoods(neighborhood_indices: NeighborhoodInput) -> bool:
    """Return True when neighborhood_indices is a non-object 2D NumPy array."""
    return (
        isinstance(neighborhood_indices, np.ndarray)
        and neighborhood_indices.ndim == 2
        and neighborhood_indices.dtype != object
    )


def pad_neighborhoods(
    neighborhood_indices: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert variable-length neighbor sequences into padded indices, mask, and counts.

    Args:
        neighborhood_indices: Sequence of length N where each entry holds neighbor indices.

    Returns:
        padded_indices: Integer array of shape (N, max_k) padded with 0.
        mask: Boolean array of shape (N, max_k), True for valid neighbor entries.
        counts: Integer array of shape (N,) with the neighbor count for each point.
    """
    n_points = len(neighborhood_indices)
    if n_points == 0:
        return (
            np.empty((0, 0), dtype=np.intp),
            np.empty((0, 0), dtype=bool),
            np.empty(0, dtype=np.intp),
        )

    counts = np.array([len(neighbors) for neighbors in neighborhood_indices], dtype=np.intp)
    if np.any(counts == 0):
        raise ValueError("Every point must have at least one neighbor.")

    max_k = int(np.max(counts))
    padded_indices = np.zeros((n_points, max_k), dtype=np.intp)
    mask = np.zeros((n_points, max_k), dtype=bool)

    for i, neighbors in enumerate(neighborhood_indices):
        k_i = int(counts[i])
        padded_indices[i, :k_i] = np.asarray(neighbors, dtype=np.intp)
        mask[i, :k_i] = True

    return padded_indices, mask, counts
