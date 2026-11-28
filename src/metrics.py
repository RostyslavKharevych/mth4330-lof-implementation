import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate_metric_inputs(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    require_both_classes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary ground-truth labels and continuous anomaly scores."""
    y_raw = np.asarray(y_true)
    scores_raw = np.asarray(scores, dtype=np.float64)

    if y_raw.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_raw.shape}.")
    if scores_raw.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {scores_raw.shape}.")
    if y_raw.shape[0] == 0:
        raise ValueError("y_true and scores must not be empty.")
    if y_raw.shape[0] != scores_raw.shape[0]:
        raise ValueError(
            f"Length mismatch between y_true ({y_raw.shape[0]}) and scores ({scores_raw.shape[0]})."
        )
    if not np.all(np.isfinite(scores_raw)):
        raise ValueError("scores must contain only finite values (no NaN or Inf).")
    if not np.all((y_raw == 0) | (y_raw == 1)):
        raise ValueError("y_true must contain only binary labels in {0, 1}.")

    y_arr = y_raw.astype(np.int64, copy=False)
    n_anomalies = int(np.sum(y_arr))
    if n_anomalies == 0:
        raise ValueError("y_true must contain at least one anomaly (label 1).")
    if require_both_classes and n_anomalies == len(y_arr):
        raise ValueError(
            "y_true must contain both normal (0) and anomaly (1) samples for ROC-AUC."
        )

    return y_arr, scores_raw


def compute_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Average Precision (area under the precision-recall curve)."""
    y_arr, scores_arr = _validate_metric_inputs(y_true, scores, require_both_classes=False)
    return float(average_precision_score(y_arr, scores_arr))


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC from binary labels and continuous anomaly scores."""
    y_arr, scores_arr = _validate_metric_inputs(y_true, scores, require_both_classes=True)
    return float(roc_auc_score(y_arr, scores_arr))


def compute_precision_at_n(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute permutation-invariant tie-averaged Precision@n.

    Here n = sum(y_true) is the exact number of ground-truth anomalies in the
    evaluation partition. When multiple samples share the n-th highest score,
    expected precision over tied boundary items is returned:
        (tp_above + tp_tied * (remaining / n_tied)) / n
    """
    y_arr, scores_arr = _validate_metric_inputs(y_true, scores, require_both_classes=False)
    n = int(np.sum(y_arr))

    cutoff = np.sort(scores_arr)[::-1][n - 1]
    above_mask = scores_arr > cutoff
    tied_mask = scores_arr == cutoff

    n_above = int(np.sum(above_mask))
    n_tied = int(np.sum(tied_mask))
    tp_above = int(np.sum(y_arr[above_mask]))
    tp_tied = int(np.sum(y_arr[tied_mask]))
    remaining = n - n_above

    return float((tp_above + tp_tied * (remaining / n_tied)) / n)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Evaluate anomaly scores using Average Precision, ROC-AUC, and Precision@n."""
    y_arr, scores_arr = _validate_metric_inputs(y_true, scores, require_both_classes=True)
    return {
        "ap": compute_average_precision(y_arr, scores_arr),
        "roc_auc": compute_roc_auc(y_arr, scores_arr),
        "precision_at_n": compute_precision_at_n(y_arr, scores_arr),
    }
