
import numpy as np
import pytest

from src.model.distances import pairwise_euclidean_distances
from src.model.neighborhoods import calculate_minpts_neighborhoods


def test_known_minpts_distances():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    D = pairwise_euclidean_distances(X)

    k_dist, neighbors = calculate_minpts_neighborhoods(D, min_pts=2)

    np.testing.assert_allclose(k_dist, [2.0, 1.0, 2.0, 9.0])
    np.testing.assert_array_equal(neighbors[0], [1, 2])
    np.testing.assert_array_equal(neighbors[1], [0, 2])
    np.testing.assert_array_equal(neighbors[2], [0, 1])
    np.testing.assert_array_equal(neighbors[3], [1, 2])


def test_ties_expand_neighborhood():
    X = np.array([[0.0], [1.0], [-1.0], [5.0]])
    D = pairwise_euclidean_distances(X)

    k_dist, neighbors = calculate_minpts_neighborhoods(D, min_pts=1)

    assert k_dist[0] == 1.0
    np.testing.assert_array_equal(neighbors[0], [1, 2])
    assert len(neighbors[0]) > 1


def test_duplicate_observations():
    X = np.array([[0.0], [0.0], [2.0]])
    D = pairwise_euclidean_distances(X)

    k_dist, neighbors = calculate_minpts_neighborhoods(D, min_pts=1)

    assert k_dist[0] == 0.0
    np.testing.assert_array_equal(neighbors[0], [1])
    np.testing.assert_array_equal(neighbors[1], [0])
    np.testing.assert_array_equal(neighbors[2], [0, 1])

    assert all(i not in indices for i, indices in enumerate(neighbors))


@pytest.mark.parametrize("min_pts", [0, -1, 3, 1.5, True])
def test_invalid_minpts(min_pts):
    D = np.zeros((3, 3))

    with pytest.raises(ValueError):
        calculate_minpts_neighborhoods(D, min_pts)


@pytest.mark.parametrize("D", [
    np.ones((2, 3)),
    np.array([[0.0, -1.0], [-1.0, 0.0]]),
    np.array([[0.0, np.nan], [np.nan, 0.0]]),
    np.array([[0.0, 1.0], [2.0, 0.0]]),
])
def test_invalid_distance_matrix(D):
    with pytest.raises(ValueError):
        calculate_minpts_neighborhoods(D, min_pts=1)
