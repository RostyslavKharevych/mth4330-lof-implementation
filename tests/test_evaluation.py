import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import LocalOutlierFactor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluator import (
    evaluate_model,
    evaluate_split,
    evaluate_splits,
    make_lof_scorer,
)
from src.metrics import (
    compute_average_precision,
    compute_precision_at_n,
    compute_roc_auc,
    evaluate_scores,
)
from src.preprocessing import generate_preprocessed_splits


def test_metrics_perfect_ranking() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.1, 0.95, 0.2, 0.85, 0.05, 0.3])

    metrics = evaluate_scores(y_true, scores)
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["precision_at_n"] == pytest.approx(1.0)


def test_metrics_inverted_ranking() -> None:
    y_true = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

    metrics = evaluate_scores(y_true, scores)
    assert metrics["precision_at_n"] == pytest.approx(0.0)
    assert metrics["roc_auc"] == pytest.approx(0.0)
    assert metrics["ap"] < 0.5


def test_precision_at_n_tie_handling_is_permutation_invariant() -> None:
    # Partial tie crossing cutoff (n=2): one positive strictly above, one positive + one negative tied at 0.5
    y_partial = np.array([1, 1, 0, 0])
    scores_partial = np.array([0.9, 0.5, 0.5, 0.1])
    assert compute_precision_at_n(y_partial, scores_partial) == pytest.approx(0.75)

    # Reordered rows must yield identical tie-averaged Precision@n
    perm = np.array([2, 1, 3, 0])
    assert compute_precision_at_n(y_partial[perm], scores_partial[perm]) == pytest.approx(0.75)

    # All scores tied must equal base anomaly rate regardless of label ordering
    y_all_tied_a = np.array([1, 1, 0, 0, 0])
    y_all_tied_b = np.array([0, 0, 0, 1, 1])
    constant_scores = np.full(5, 1.0)
    assert compute_precision_at_n(y_all_tied_a, constant_scores) == pytest.approx(0.4)
    assert compute_precision_at_n(y_all_tied_b, constant_scores) == pytest.approx(0.4)


def test_metrics_validation_errors() -> None:
    # Zero anomalies
    y_no_anomalies = np.zeros(5, dtype=np.int64)
    scores = np.linspace(0.1, 0.5, 5)
    with pytest.raises(ValueError, match="at least one anomaly"):
        compute_average_precision(y_no_anomalies, scores)
    with pytest.raises(ValueError, match="at least one anomaly"):
        compute_roc_auc(y_no_anomalies, scores)
    with pytest.raises(ValueError, match="at least one anomaly"):
        compute_precision_at_n(y_no_anomalies, scores)
    with pytest.raises(ValueError, match="at least one anomaly"):
        evaluate_scores(y_no_anomalies, scores)

    # All anomalies -> ROC-AUC undefined
    y_all_anomalies = np.ones(4, dtype=np.int64)
    assert compute_precision_at_n(y_all_anomalies, np.ones(4)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="both normal"):
        compute_roc_auc(y_all_anomalies, np.arange(4, dtype=float))
    with pytest.raises(ValueError, match="both normal"):
        evaluate_scores(y_all_anomalies, np.arange(4, dtype=float))

    # Mismatched length, empty, non-binary labels, non-finite scores
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_scores(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="not be empty"):
        evaluate_scores(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="binary labels"):
        evaluate_scores(np.array([0, 2, 1]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="binary labels"):
        evaluate_scores(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_scores(np.array([0, 1, 0]), np.array([0.1, np.nan, 0.3]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_scores(np.array([0, 1, 0]), np.array([0.1, np.inf, 0.3]))


def test_evaluator_unified_interface_and_label_withholding() -> None:
    rng = np.random.default_rng(42)
    X_normal = rng.normal(loc=0.0, scale=0.4, size=(90, 2))
    X_outliers = np.array([[6.0, 6.0], [-6.0, 5.5], [5.5, -6.0], [-5.5, -5.5]])
    X = np.vstack([X_normal, X_outliers])
    y = np.concatenate([np.zeros(len(X_normal), dtype=np.int64), np.ones(len(X_outliers), dtype=np.int64)])

    splits = generate_preprocessed_splits(X, y, seeds=(42, 123), scale=True)
    split = splits[42]

    # 1. Transductive LOF scorer evaluates val and test partitions cleanly
    lof_scorer = make_lof_scorer(min_pts=5)
    val_metrics = evaluate_split(lof_scorer, split, partition="val")
    test_metrics = evaluate_split(lof_scorer, split, partition="test")

    for metrics in (val_metrics, test_metrics):
        assert set(metrics.keys()) == {"ap", "roc_auc", "precision_at_n"}
        assert metrics["ap"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["precision_at_n"] == pytest.approx(1.0)

    # 2. Inductive baseline scorer shares the exact same (X_train, X_eval) interface
    seen_args: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def centroid_distance_baseline(X_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
        seen_args.append((X_train.shape, X_eval.shape))
        center = np.mean(X_train, axis=0)
        return np.linalg.norm(X_eval - center, axis=1)

    baseline_val = evaluate_model(centroid_distance_baseline, split.X_train, split.X_val, split.y_val)
    assert seen_args == [(split.X_train.shape, split.X_val.shape)]
    assert baseline_val["roc_auc"] == pytest.approx(1.0)

    # 3. Multi-seed aggregation across splits
    aggregated = evaluate_splits(lof_scorer, splits, partition="test")
    assert set(aggregated.keys()) == {"per_seed", "mean", "std"}
    assert set(aggregated["per_seed"].keys()) == {42, 123}
    assert aggregated["mean"]["roc_auc"] == pytest.approx(1.0)
    assert aggregated["std"]["roc_auc"] == pytest.approx(0.0)


def test_evaluator_custom_graph_builder_one_line_swap() -> None:
    calls: list[int] = []

    def custom_graph_builder(X: np.ndarray, min_pts: int) -> tuple[np.ndarray, np.ndarray]:
        calls.append(min_pts)
        n = X.shape[0]
        # Fixed dummy neighborhoods & reachability where last point is an obvious outlier
        neighborhoods = np.tile(np.arange(min_pts), (n, 1))
        reach = np.ones((n, n), dtype=np.float64)
        reach[-1, :] = 100.0
        return neighborhoods, reach

    scorer = make_lof_scorer(min_pts=3, graph_builder=custom_graph_builder)
    X_dummy = np.zeros((10, 2))
    y_dummy = np.array([0] * 9 + [1], dtype=np.int64)

    metrics = evaluate_model(scorer, X_dummy, X_dummy, y_dummy)
    assert calls == [3]
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["precision_at_n"] == pytest.approx(1.0)


def test_evaluator_validation_errors() -> None:
    splits = generate_preprocessed_splits(
        np.random.default_rng(0).normal(size=(20, 2)),
        np.array([0] * 15 + [1] * 5, dtype=np.int64),
        seeds=(42,),
    )
    scorer = make_lof_scorer(min_pts=2)

    with pytest.raises(ValueError, match="partition must be 'val' or 'test'"):
        evaluate_split(scorer, splits[42], partition="train")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_splits(scorer, {})

    def bad_shape_scorer(_X_train: np.ndarray, _X_eval: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    with pytest.raises(ValueError, match="1D scores of shape"):
        evaluate_split(bad_shape_scorer, splits[42], partition="val")

    with pytest.raises(ValueError, match="integer >= 1"):
        make_lof_scorer(min_pts=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="integer >= 1"):
        make_lof_scorer(min_pts=2.5)  # type: ignore[arg-type]


def test_evaluator_upstream_pipeline_parity_with_sklearn() -> None:
    rng = np.random.default_rng(7)
    dense = rng.normal(loc=(0.0, 0.0), scale=0.2, size=(40, 2))
    sparse = rng.normal(loc=(12.0, 12.0), scale=2.0, size=(40, 2))
    outliers = np.array([[2.0, 2.0], [-2.0, 1.8], [22.0, 22.0]])
    X_eval = np.vstack([dense, sparse, outliers])

    min_pts = 8
    scorer = make_lof_scorer(min_pts=min_pts)
    our_scores = scorer(np.empty((0, 2)), X_eval)

    sklearn_model = LocalOutlierFactor(n_neighbors=min_pts, metric="euclidean")
    sklearn_model.fit(X_eval)
    sklearn_scores = -sklearn_model.negative_outlier_factor_

    assert float(np.max(np.abs(our_scores - sklearn_scores))) <= 1e-10
    assert float(pearsonr(our_scores, sklearn_scores).statistic) >= 0.999
    assert float(spearmanr(our_scores, sklearn_scores).statistic) >= 0.999
