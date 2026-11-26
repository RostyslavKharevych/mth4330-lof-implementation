import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import LocalOutlierFactor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.lfc import calculate_lrd_loops, calculate_lrd_vectorized
from src.model.lof import calculate_lof_iterative, calculate_lof_vectorized


def compute_temporary_lof_inputs(
    X: np.ndarray, min_pts: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Temporary helper computing pairwise distances, k-distance,
    boundary-tie-inclusive neighborhoods, and reachability distances.

    Args:
        X: Data matrix of shape (N, d).
        min_pts: Neighborhood size parameter (MinPts), 1 <= min_pts < N.

    Returns:
        D: Pairwise Euclidean distance matrix of shape (N, N).
        kdist: k-distance array of shape (N,).
        neighborhoods: List of N 1D integer arrays where
                       N[i] = {j != i : D[i, j] <= kdist[i]}.
        reach_dist: Reachability distance matrix of shape (N, N) where
                    reach_dist[p, o] = max(kdist[o], D[p, o]).
    """
    n_points = X.shape[0]
    if not (1 <= min_pts < n_points):
        raise ValueError(f"min_pts must satisfy 1 <= min_pts < N (got min_pts={min_pts}, N={n_points}).")

    D = cdist(X, X, metric="euclidean")
    np.fill_diagonal(D, 0.0)

    kdist = np.empty(n_points, dtype=float)
    neighborhoods: list[np.ndarray] = []

    all_indices = np.arange(n_points)
    for i in range(n_points):
        other_mask = all_indices != i
        other_indices = all_indices[other_mask]
        other_dists = D[i, other_mask]

        sorted_dists = np.sort(other_dists)
        k_distance_i = sorted_dists[min_pts - 1]
        kdist[i] = k_distance_i

        neighbor_indices = other_indices[other_dists <= k_distance_i]
        # Order neighbors deterministically by distance then index
        order = np.lexsort((neighbor_indices, D[i, neighbor_indices]))
        neighborhoods.append(neighbor_indices[order])

    reach_dist = np.maximum(kdist[np.newaxis, :], D)
    return D, kdist, neighborhoods, reach_dist


def _make_varying_density_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic 2D dataset with a dense cluster, a sparse cluster,
    and injected local/global outliers without distance ties.
    """
    rng = np.random.default_rng(seed)

    dense_cluster = rng.normal(loc=(0.0, 0.0), scale=0.15, size=(60, 2))
    sparse_cluster = rng.normal(loc=(15.0, 15.0), scale=2.5, size=(60, 2))

    # Local outliers near the dense cluster + distant outliers
    outliers = np.array(
        [
            [1.6, 1.4],
            [-1.5, 1.3],
            [1.8, -1.2],
            [27.5, 26.0],
            [-7.5, 19.0],
        ],
        dtype=float,
    )

    X = np.vstack([dense_cluster, sparse_cluster, outliers])
    n_inliers = len(dense_cluster) + len(sparse_cluster)
    inlier_mask = np.zeros(len(X), dtype=bool)
    inlier_mask[:n_inliers] = True
    outlier_mask = ~inlier_mask
    return X, inlier_mask, outlier_mask


def _make_tied_dataset(seed: int = 7) -> np.ndarray:
    """
    Create a varying-density dataset with localized boundary distance ties
    plus injected outliers.
    """
    rng = np.random.default_rng(seed)

    # Isolated small square sub-cluster creates exact boundary ties at kdist for min_pts=3
    square = np.array(
        [
            [-6.0, -6.0],
            [-5.0, -6.0],
            [-6.0, -5.0],
            [-5.0, -5.0],
            [-5.5, -5.5],
        ],
        dtype=float,
    )
    dense_cluster = rng.normal(loc=(0.5, 0.5), scale=0.3, size=(45, 2))
    sparse_cluster = rng.normal(loc=(20.0, 20.0), scale=2.2, size=(45, 2))
    outliers = np.array(
        [
            [5.0, 5.0],
            [-4.0, 3.5],
            [31.0, 30.0],
        ],
        dtype=float,
    )
    return np.vstack([square, dense_cluster, sparse_cluster, outliers])


def _make_lattice_dataset(seed: int = 11) -> tuple[np.ndarray, int]:
    """
    Create a 6x6 integer lattice cluster alongside a sparse cluster and outliers,
    producing widespread boundary distance ties.
    """
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(6, dtype=float), np.arange(6, dtype=float))
    grid_cluster = np.column_stack([gx.ravel(), gy.ravel()])
    sparse_cluster = rng.normal(loc=(20.0, 20.0), scale=2.0, size=(35, 2))
    outliers = np.array(
        [
            [8.0, 8.0],
            [-3.5, 2.5],
            [30.0, 30.0],
        ],
        dtype=float,
    )
    X = np.vstack([grid_cluster, sparse_cluster, outliers])
    return X, len(outliers)


@pytest.mark.parametrize("min_pts", [5, 10, 20])
def test_tie_free_parity_with_sklearn(min_pts: int) -> None:
    X, inlier_mask, outlier_mask = _make_varying_density_dataset()
    _, _, neighborhoods, reach_dist = compute_temporary_lof_inputs(X, min_pts)

    # Confirm no distance ties occurred so neighborhood size equals min_pts everywhere
    assert all(len(neighbors) == min_pts for neighbors in neighborhoods)
    rect_neighborhoods = np.vstack(neighborhoods)

    sklearn_model = LocalOutlierFactor(n_neighbors=min_pts, metric="euclidean")
    sklearn_model.fit(X)
    sklearn_lrd = sklearn_model._lrd
    sklearn_lof = -sklearn_model.negative_outlier_factor_

    for neighborhood_arg in (rect_neighborhoods, neighborhoods):
        lrd_loops = calculate_lrd_loops(neighborhood_arg, reach_dist)
        lrd_vec = calculate_lrd_vectorized(neighborhood_arg, reach_dist)
        assert np.allclose(lrd_loops, lrd_vec, atol=1e-12, rtol=1e-12)
        assert np.allclose(lrd_vec, sklearn_lrd, atol=1e-10, rtol=0.0)

        lof_loops = calculate_lof_iterative(neighborhood_arg, lrd_loops)
        lof_vec = calculate_lof_vectorized(neighborhood_arg, lrd_vec)
        assert np.allclose(lof_loops, lof_vec, atol=1e-12, rtol=1e-12)

        max_abs_diff = float(np.max(np.abs(lof_vec - sklearn_lof)))
        assert max_abs_diff <= 1e-10

        pearson_corr = float(pearsonr(lof_vec, sklearn_lof).statistic)
        spearman_corr = float(spearmanr(lof_vec, sklearn_lof).statistic)
        assert pearson_corr >= 0.999
        assert spearman_corr >= 0.999

        # Injected local/global outliers should score distinctly higher than cluster cores
        assert np.min(lof_vec[outlier_mask]) > np.median(lof_vec[inlier_mask])
        assert np.min(lof_vec[outlier_mask]) > 1.5


def test_boundary_ties_correlation_with_sklearn() -> None:
    min_pts = 3
    X = _make_tied_dataset()
    _, _, neighborhoods, reach_dist = compute_temporary_lof_inputs(X, min_pts)

    # Confirm boundary ties produced variable-length neighborhoods > min_pts
    assert any(len(neighbors) > min_pts for neighbors in neighborhoods)

    lrd_loops = calculate_lrd_loops(neighborhoods, reach_dist)
    lrd_vec = calculate_lrd_vectorized(neighborhoods, reach_dist)
    assert np.allclose(lrd_loops, lrd_vec, atol=1e-12, rtol=1e-12)

    lof_loops = calculate_lof_iterative(neighborhoods, lrd_loops)
    lof_vec = calculate_lof_vectorized(neighborhoods, lrd_vec)
    assert np.allclose(lof_loops, lof_vec, atol=1e-12, rtol=1e-12)

    sklearn_model = LocalOutlierFactor(n_neighbors=min_pts, metric="euclidean")
    sklearn_model.fit(X)
    sklearn_lof = -sklearn_model.negative_outlier_factor_

    pearson_corr = float(pearsonr(lof_vec, sklearn_lof).statistic)
    spearman_corr = float(spearmanr(lof_vec, sklearn_lof).statistic)
    assert pearson_corr >= 0.999
    assert spearman_corr >= 0.999


def test_lattice_ties_outlier_ranking() -> None:
    min_pts = 4
    X, n_outliers = _make_lattice_dataset()
    _, _, neighborhoods, reach_dist = compute_temporary_lof_inputs(X, min_pts)

    assert any(len(neighbors) > min_pts for neighbors in neighborhoods)

    lrd_loops = calculate_lrd_loops(neighborhoods, reach_dist)
    lrd_vec = calculate_lrd_vectorized(neighborhoods, reach_dist)
    assert np.allclose(lrd_loops, lrd_vec, atol=1e-12, rtol=1e-12)

    lof_loops = calculate_lof_iterative(neighborhoods, lrd_loops)
    lof_vec = calculate_lof_vectorized(neighborhoods, lrd_vec)
    assert np.allclose(lof_loops, lof_vec, atol=1e-12, rtol=1e-12)

    sklearn_model = LocalOutlierFactor(n_neighbors=min_pts, metric="euclidean")
    sklearn_model.fit(X)
    sklearn_lof = -sklearn_model.negative_outlier_factor_

    assert float(pearsonr(lof_vec, sklearn_lof).statistic) >= 0.999
    expected_outlier_indices = set(range(len(X) - n_outliers, len(X)))
    assert set(np.argsort(lof_vec)[-n_outliers:]) == expected_outlier_indices


def test_duplicate_coordinates_zero_reachability_edge_case() -> None:
    """
    When > MinPts points share identical coordinates, kdist == 0 and reach_dist == 0
    within that cluster (S[i] == 0 and S[j] == 0 for all neighbors j).
    Epsilon stabilization must prevent division by zero and enforce LOF(p) == 1.0.
    """
    min_pts = 4
    n_duplicates = 7  # strictly > min_pts + 1 so k-neighbors are all identical

    rng = np.random.default_rng(123)
    duplicates = np.tile([0.0, 0.0], (n_duplicates, 1))
    other_cluster = rng.normal(loc=(10.0, 10.0), scale=0.5, size=(25, 2))
    X = np.vstack([duplicates, other_cluster])

    _, kdist, neighborhoods, reach_dist = compute_temporary_lof_inputs(X, min_pts)

    for i in range(n_duplicates):
        assert kdist[i] == 0.0
        assert len(neighborhoods[i]) == n_duplicates - 1
        sum_reach_i = float(np.sum(reach_dist[i, neighborhoods[i]]))
        assert sum_reach_i == 0.0
        for j in neighborhoods[i]:
            assert float(np.sum(reach_dist[j, neighborhoods[j]])) == 0.0

    epsilon = 1e-10
    lrd_loops = calculate_lrd_loops(neighborhoods, reach_dist, epsilon=epsilon)
    lrd_vec = calculate_lrd_vectorized(neighborhoods, reach_dist, epsilon=epsilon)
    assert np.all(np.isfinite(lrd_loops))
    assert np.all(np.isfinite(lrd_vec))
    assert np.allclose(lrd_loops[:n_duplicates], 1.0 / epsilon, atol=0.0, rtol=1e-12)
    assert np.allclose(lrd_vec[:n_duplicates], 1.0 / epsilon, atol=0.0, rtol=1e-12)

    lof_loops = calculate_lof_iterative(neighborhoods, lrd_loops)
    lof_vec = calculate_lof_vectorized(neighborhoods, lrd_vec)
    assert np.all(np.isfinite(lof_loops))
    assert np.all(np.isfinite(lof_vec))
    assert np.allclose(lof_loops[:n_duplicates], 1.0, atol=1e-12, rtol=0.0)
    assert np.allclose(lof_vec[:n_duplicates], 1.0, atol=1e-12, rtol=0.0)


def test_all_identical_coordinates_rectangular_and_ragged() -> None:
    min_pts = 4
    n_points = 8
    X = np.tile([3.25, -1.5], (n_points, 1))

    _, kdist, neighborhoods, reach_dist = compute_temporary_lof_inputs(X, min_pts)
    assert np.all(kdist == 0.0)
    assert np.all(reach_dist == 0.0)

    rect_neighborhoods = np.vstack(neighborhoods)
    assert rect_neighborhoods.shape == (n_points, n_points - 1)

    epsilon = 1e-10
    for neighborhood_arg in (rect_neighborhoods, neighborhoods):
        lrd_loops = calculate_lrd_loops(neighborhood_arg, reach_dist, epsilon=epsilon)
        lrd_vec = calculate_lrd_vectorized(neighborhood_arg, reach_dist, epsilon=epsilon)
        assert np.allclose(lrd_loops, 1.0 / epsilon, atol=0.0, rtol=1e-12)
        assert np.allclose(lrd_vec, 1.0 / epsilon, atol=0.0, rtol=1e-12)

        lof_loops = calculate_lof_iterative(neighborhood_arg, lrd_loops)
        lof_vec = calculate_lof_vectorized(neighborhood_arg, lrd_vec)
        assert np.allclose(lof_loops, 1.0, atol=1e-12, rtol=0.0)
        assert np.allclose(lof_vec, 1.0, atol=1e-12, rtol=0.0)


def test_empty_and_zero_neighbor_validation() -> None:
    empty_rect = np.empty((0, 3), dtype=np.intp)
    empty_reach = np.empty((0, 0), dtype=float)
    empty_lrd = np.empty(0, dtype=float)

    assert calculate_lrd_loops(empty_rect, empty_reach).shape == (0,)
    assert calculate_lrd_vectorized(empty_rect, empty_reach).shape == (0,)
    assert calculate_lof_iterative(empty_rect, empty_lrd).shape == (0,)
    assert calculate_lof_vectorized(empty_rect, empty_lrd).shape == (0,)

    with pytest.raises(ValueError):
        calculate_lrd_vectorized([np.array([1]), np.array([], dtype=np.intp)], np.zeros((2, 2)))

    with pytest.raises(ValueError):
        calculate_lof_vectorized(np.empty((2, 0), dtype=np.intp), np.ones(2))
