
import numpy as np
import pytest

from src.model.distances import pairwise_euclidean_distances
from src.model.neighborhoods import calculate_minpts_neighborhoods
from src.model.reachability import calculate_reachability_distances
from src.model.lfc import calculate_lrd_vectorized
from src.model.lof import calculate_lof_vectorized


def test_known_reachability_distances():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])

    D = pairwise_euclidean_distances(X)
    k_dist, _ = calculate_minpts_neighborhoods(D, min_pts=2)

    reach = calculate_reachability_distances(D, k_dist)

    np.testing.assert_allclose(k_dist, [3, 2, 3, 9])

    expected = np.array([
        [3, 2, 3, 10],
        [3, 2, 3,  9],
        [3, 2, 3,  9],
        [10, 9, 7, 9]
    ])

    np.testing.assert_allclose(reach, expected)


def test_reachability_is_directional():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    D = pairwise_euclidean_distances(X)
    k_dist, _ = calculate_minpts_neighborhoods(D, min_pts=2)

    reach = calculate_reachability_distances(D, k_dist)

    assert reach[0, 1] == 2.0
    assert reach[1, 0] == 3.0


def test_end_to_end_lof_integration():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])

    D = pairwise_euclidean_distances(X)

    k_dist, neighbors = calculate_minpts_neighborhoods(D, min_pts=2)

    reach = calculate_reachability_distances(D, k_dist)

    lrd = calculate_lrd_vectorized(neighbors, reach)
    lof = calculate_lof_vectorized(neighbors, lrd)

    assert lof.shape == (4,)
    assert np.all(np.isfinite(lof))

    np.testing.assert_allclose(
        lrd, [0.4, 1 / 3, 0.4, 0.125], rtol=1e-8
    )
    np.testing.assert_allclose(
        lof, [11 / 12, 1.2, 11 / 12, 44 / 15], rtol=1e-8
    )

    # Observation at 10 is the strongest local outlier.
    assert np.argmax(lof) == 3


@pytest.mark.parametrize("D,k_dist", [
    (np.ones((2, 3)), np.ones(2)),
    (np.eye(3), np.ones(2)),
    (np.array([[0, -1], [-1, 0]]), np.ones(2)),
    (np.array([[0, np.nan], [np.nan, 0]]), np.ones(2)),
    (np.array([[0, 1], [2, 0]]), np.ones(2)),
])
def test_invalid_reachability_inputs(D, k_dist):
    with pytest.raises(ValueError):
        calculate_reachability_distances(D, k_dist)
