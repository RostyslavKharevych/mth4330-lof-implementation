
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.model.distances import (
    validate_input,
    pairwise_euclidean_distances,
)


def test_known_euclidean_distances():
    X = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0],
    ])

    expected = np.array([
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 5.0],
        [10.0, 5.0, 0.0],
    ])

    actual = pairwise_euclidean_distances(X)
    np.testing.assert_allclose(actual, expected)


def test_matches_scipy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 6))

    actual = pairwise_euclidean_distances(X)
    expected = cdist(X, X, metric="euclidean")

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_distance_matrix_properties():
    X = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
        [5.0, 6.0],
    ])

    D = pairwise_euclidean_distances(X)

    assert D.shape == (3, 3)
    assert np.all(D >= 0)
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(np.diag(D), np.zeros(3))

    # Two distinct observations may have identical coordinates.
    assert D[0, 1] == 0.0


@pytest.mark.parametrize("X", [
    np.array([1.0, 2.0]),
    np.empty((0, 3)),
    np.empty((3, 0)),
    np.array([[1.0, np.nan]]),
    np.array([[1.0, np.inf]]),
])
def test_invalid_input(X):
    with pytest.raises(ValueError):
        validate_input(X)
