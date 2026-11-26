from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1011)


@dataclass(frozen=True)
class DatasetSplit:
    """Container holding train/validation/test feature matrices, labels, seed, and fitted scaler."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    seed: int
    scaler: Optional[StandardScaler] = None


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> DatasetSplit:
    """
    Create a deterministic stratified 60/20/20 train/validation/test split.

    Args:
        X: Feature matrix of shape (N, d).
        y: Binary label array of shape (N,).
        random_state: Seed controlling both stratified split stages.
        train_ratio: Fraction allocated to the training set.
        val_ratio: Fraction allocated to the validation set.
        test_ratio: Fraction allocated to the test set.

    Returns:
        DatasetSplit with unscaled float64 features and int64 labels.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).ravel()

    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Mismatched or invalid input shapes: X={X_arr.shape}, y={y_arr.shape}."
        )
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}."
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_arr,
        y_arr,
        train_size=train_ratio,
        stratify=y_arr,
        random_state=random_state,
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_fraction_of_temp,
        stratify=y_temp,
        random_state=random_state,
    )

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        seed=random_state,
        scaler=None,
    )


def standardize_split(split: DatasetSplit) -> DatasetSplit:
    """
    Standardize train, validation, and test features without data leakage.

    Fits StandardScaler strictly on split.X_train and transforms all three
    partitions using only training mean and standard deviation. Zero-variance
    columns in X_train keep scale_ = 1.0 so no division by zero occurs.
    """
    scaler = StandardScaler()
    scaler.fit(split.X_train)

    # Explicitly guard zero-variance columns (StandardScaler also sets 0 -> 1.0)
    if scaler.scale_ is not None:
        scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)

    X_train_scaled = np.asarray(scaler.transform(split.X_train), dtype=np.float64)
    X_val_scaled = np.asarray(scaler.transform(split.X_val), dtype=np.float64)
    X_test_scaled = np.asarray(scaler.transform(split.X_test), dtype=np.float64)

    return DatasetSplit(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
        seed=split.seed,
        scaler=scaler,
    )


def generate_preprocessed_splits(
    X: np.ndarray,
    y: np.ndarray,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scale: bool = True,
) -> dict[int, DatasetSplit]:
    """
    Generate deterministic 60/20/20 stratified splits across multiple random seeds.

    Args:
        X: Feature matrix of shape (N, d).
        y: Binary label vector of shape (N,).
        seeds: Sequence of distinct integer seeds (defaults to DEFAULT_SEEDS).
        scale: When True, apply leak-free training-only standardization to each split.

    Returns:
        Dictionary mapping each seed to its DatasetSplit.
    """
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one random seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {list(seeds)}.")

    splits: dict[int, DatasetSplit] = {}
    for seed in seeds:
        split = stratified_train_val_test_split(X, y, random_state=seed)
        splits[seed] = standardize_split(split) if scale else split
    return splits
