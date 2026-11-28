from typing import Any, Callable, Literal, Mapping

import numpy as np

from .metrics import evaluate_scores
from .model._utils import NeighborhoodInput
from .model.distances import pairwise_euclidean_distances
from .model.lrd import calculate_lrd_vectorized
from .model.lof import calculate_lof_vectorized
from .model.neighborhoods import calculate_minpts_neighborhoods
from .model.reachability import calculate_reachability_distances
from .preprocessing import DatasetSplit

ScorerFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
NeighborhoodBuilder = Callable[[np.ndarray, int], tuple[NeighborhoodInput, np.ndarray]]


def upstream_lof_graph_builder(
    X: np.ndarray, min_pts: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Build tie-inclusive MinPts neighborhoods and reachability distances using the
    upstream distances, neighborhoods, and reachability modules.
    """
    D = pairwise_euclidean_distances(X)
    kdist, neighborhoods = calculate_minpts_neighborhoods(D, min_pts=min_pts)
    reach_dist = calculate_reachability_distances(D, kdist)
    return neighborhoods, reach_dist


def make_lof_scorer(
    min_pts: int,
    graph_builder: NeighborhoodBuilder = upstream_lof_graph_builder,
    lrd_fn: Callable[[NeighborhoodInput, np.ndarray, float], np.ndarray] = calculate_lrd_vectorized,
    lof_fn: Callable[[NeighborhoodInput, np.ndarray], np.ndarray] = calculate_lof_vectorized,
    epsilon: float = 1e-10,
) -> ScorerFn:
    """
    Create a transductive LOF scorer conforming to the unified (X_train, X_eval) interface.

    Computes LOF scores directly on X_eval without referencing ground-truth labels.
    Pass `graph_builder=...` to swap the neighborhood/reachability backend.
    """
    if isinstance(min_pts, (bool, np.bool_)) or not isinstance(min_pts, (int, np.integer)):
        raise ValueError(f"min_pts must be an integer >= 1, got {min_pts!r}.")
    if min_pts < 1:
        raise ValueError(f"min_pts must be >= 1, got {min_pts}.")

    def _score(_X_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
        X_eval_arr = np.asarray(X_eval, dtype=np.float64)
        if X_eval_arr.ndim != 2:
            raise ValueError(f"X_eval must be 2D, got shape {X_eval_arr.shape}.")
        if X_eval_arr.shape[0] <= min_pts:
            raise ValueError(
                f"X_eval must have more than min_pts ({min_pts}) samples, got {X_eval_arr.shape[0]}."
            )

        neighborhoods, reach_dist = graph_builder(X_eval_arr, int(min_pts))
        lrd = lrd_fn(neighborhoods, reach_dist, epsilon)
        lof_scores = lof_fn(neighborhoods, lrd)
        return np.asarray(lof_scores, dtype=np.float64)

    return _score


def evaluate_model(
    scorer: ScorerFn,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    """
    Score X_eval without passing y_eval to the scorer, then compute evaluation metrics.
    """
    X_train_arr = np.asarray(X_train, dtype=np.float64)
    X_eval_arr = np.asarray(X_eval, dtype=np.float64)
    y_eval_arr = np.asarray(y_eval)

    scores = np.asarray(scorer(X_train_arr, X_eval_arr), dtype=np.float64)
    if scores.shape != (X_eval_arr.shape[0],):
        raise ValueError(
            f"Scorer must return 1D scores of shape ({X_eval_arr.shape[0]},), got {scores.shape}."
        )
    return evaluate_scores(y_eval_arr, scores)


def evaluate_split(
    scorer: ScorerFn,
    split: DatasetSplit,
    partition: Literal["val", "test"] = "val",
) -> dict[str, float]:
    """Evaluate a scorer on either the validation or test partition of a DatasetSplit."""
    if partition == "val":
        return evaluate_model(scorer, split.X_train, split.X_val, split.y_val)
    if partition == "test":
        return evaluate_model(scorer, split.X_train, split.X_test, split.y_test)
    raise ValueError(f"partition must be 'val' or 'test', got '{partition}'.")


def evaluate_splits(
    scorer: ScorerFn,
    splits: Mapping[int, DatasetSplit],
    partition: Literal["val", "test"] = "test",
) -> dict[str, Any]:
    """
    Evaluate a scorer across multiple seeded DatasetSplits and aggregate mean/std.
    """
    if not splits:
        raise ValueError("splits mapping must not be empty.")

    per_seed = {
        seed: evaluate_split(scorer, split, partition=partition)
        for seed, split in splits.items()
    }
    metric_names = tuple(next(iter(per_seed.values())).keys())
    mean_metrics = {
        metric: float(np.mean([res[metric] for res in per_seed.values()]))
        for metric in metric_names
    }
    std_metrics = {
        metric: float(np.std([res[metric] for res in per_seed.values()]))
        for metric in metric_names
    }
    return {
        "per_seed": per_seed,
        "mean": mean_metrics,
        "std": std_metrics,
    }
