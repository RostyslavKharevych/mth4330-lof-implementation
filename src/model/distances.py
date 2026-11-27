
import numpy as np


def validate_input(X: np.ndarray) -> np.ndarray:
    """
    Validate a feature matrix for pairwise Euclidean distance calculation.

    Returns:
        A two-dimensional NumPy array with dtype float64.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Input must contain numerical values.") from exc

    if X.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Input must contain observations and features.")

    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or infinite values.")

    return X


def pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between observations.

    Args:
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    X = validate_input(X)

    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)

    for i in range(n_samples - 1):
        differences = X[i + 1:] - X[i]

        row_distances = np.linalg.norm(differences, axis=1)

        distances[i, i + 1:] = row_distances
        distances[i + 1:, i] = row_distances

    return distances
