import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import (
    load_cardio,
    load_creditcard,
    load_dataset,
    load_thyroid,
)
from src.preprocessing import (
    DEFAULT_SEEDS,
    DatasetSplit,
    generate_preprocessed_splits,
    standardize_split,
    stratified_train_val_test_split,
)


@pytest.mark.parametrize(
    ("name", "expected_shape", "expected_anomalies"),
    [
        ("thyroid", (3772, 6), 93),
        ("cardio", (1831, 21), 176),
        ("creditcard", (3_000, 29), 492),
    ],
)
def test_dataset_loaders_shapes_dtypes_and_anomalies(
    name: str, expected_shape: tuple[int, int], expected_anomalies: int
) -> None:
    X, y = load_dataset(name)

    assert X.shape == expected_shape
    assert y.shape == (expected_shape[0],)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert set(np.unique(y).tolist()) == {0, 1}
    assert int(np.sum(y == 1)) == expected_anomalies
    assert np.all(np.isfinite(X))


def test_creditcard_loader_determinism_and_shuffling() -> None:
    X_first, y_first = load_creditcard()
    X_second, y_second = load_creditcard()

    assert np.array_equal(X_first, X_second)
    assert np.array_equal(y_first, y_second)
    # Confirm fraud rows are shuffled across the subsample rather than grouped at start
    assert int(np.sum(y_first[:492] == 1)) < 492


def test_missing_dataset_and_unknown_name_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="thyroid.mat"):
        load_thyroid(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="6_cardio.npz"):
        load_cardio(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="creditcard.csv"):
        load_creditcard(data_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("unknown_dataset")


@pytest.mark.parametrize("dataset_name", ["thyroid", "cardio", "creditcard"])
def test_five_seed_stratified_splits_and_leak_free_scaling(dataset_name: str) -> None:
    X, y = load_dataset(dataset_name)
    full_anomaly_rate = float(np.mean(y == 1))
    n_total = len(y)

    unscaled_splits = generate_preprocessed_splits(X, y, seeds=DEFAULT_SEEDS, scale=False)
    scaled_splits = generate_preprocessed_splits(X, y, seeds=DEFAULT_SEEDS, scale=True)

    assert list(scaled_splits.keys()) == list(DEFAULT_SEEDS)
    assert len(DEFAULT_SEEDS) == 5

    train_feature_signatures = set()

    for seed in DEFAULT_SEEDS:
        raw_split = unscaled_splits[seed]
        scaled_split = scaled_splits[seed]

        n_train = len(raw_split.y_train)
        n_val = len(raw_split.y_val)
        n_test = len(raw_split.y_test)

        # Exact 60 / 20 / 20 partition sizes
        assert n_train + n_val + n_test == n_total
        assert abs((n_train / n_total) - 0.60) <= 1.0 / n_total
        assert abs((n_val / n_total) - 0.20) <= 1.0 / n_total
        assert abs((n_test / n_total) - 0.20) <= 1.0 / n_total

        # Total anomaly count preserved across partitions
        total_split_anomalies = int(
            np.sum(raw_split.y_train) + np.sum(raw_split.y_val) + np.sum(raw_split.y_test)
        )
        assert total_split_anomalies == int(np.sum(y))

        # Stratified class proportions preserved within rounding of 1 sample
        for y_part in (raw_split.y_train, raw_split.y_val, raw_split.y_test):
            part_rate = float(np.mean(y_part == 1))
            assert abs(part_rate - full_anomaly_rate) <= (1.0 / len(y_part)) + 1e-12

        train_feature_signatures.add(tuple(raw_split.X_train[:10, 0].tolist()))

        # Zero-leakage check: training set standardized to ~0 mean and ~1 std (for non-constant cols)
        assert scaled_split.scaler is not None
        train_means = np.mean(scaled_split.X_train, axis=0)
        train_stds = np.std(scaled_split.X_train, axis=0)
        raw_train_stds = np.std(raw_split.X_train, axis=0)
        non_constant = raw_train_stds > 0.0

        assert np.allclose(train_means, 0.0, atol=1e-12)
        assert np.allclose(train_stds[non_constant], 1.0, atol=1e-12)

        # Val and test must be transformed strictly with training mean and scale
        expected_mean = np.mean(raw_split.X_train, axis=0)
        expected_scale = np.where(raw_train_stds == 0.0, 1.0, raw_train_stds)
        assert np.allclose(scaled_split.scaler.mean_, expected_mean, atol=1e-12)
        assert np.allclose(scaled_split.scaler.scale_, expected_scale, atol=1e-12)

        expected_val = (raw_split.X_val - expected_mean) / expected_scale
        expected_test = (raw_split.X_test - expected_mean) / expected_scale
        assert np.allclose(scaled_split.X_val, expected_val, atol=1e-12)
        assert np.allclose(scaled_split.X_test, expected_test, atol=1e-12)

    # Different seeds produce distinct partitions
    assert len(train_feature_signatures) == len(DEFAULT_SEEDS)


def test_zero_leakage_on_shifted_val_and_test_distributions() -> None:
    """
    Construct splits where validation and test have intentionally shifted means/scales
    and a zero-variance column in training, confirming scaler never uses val/test stats.
    """
    rng = np.random.default_rng(99)
    X_train = np.column_stack(
        [
            rng.normal(loc=10.0, scale=2.0, size=100),
            np.full(100, 5.0),  # zero variance in train
        ]
    )
    X_val = np.column_stack(
        [
            rng.normal(loc=50.0, scale=10.0, size=40),
            np.full(40, 8.0),  # shifted constant in val
        ]
    )
    X_test = np.column_stack(
        [
            rng.normal(loc=-30.0, scale=0.5, size=40),
            np.full(40, 2.0),  # shifted constant in test
        ]
    )

    split = DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=np.zeros(100, dtype=np.int64),
        y_val=np.zeros(40, dtype=np.int64),
        y_test=np.zeros(40, dtype=np.int64),
        seed=42,
    )
    scaled = standardize_split(split)

    assert np.all(np.isfinite(scaled.X_train))
    assert np.all(np.isfinite(scaled.X_val))
    assert np.all(np.isfinite(scaled.X_test))

    # Zero-variance column in train maps to 0.0 on train and (val - 5.0) / 1.0 on val/test
    assert np.allclose(scaled.X_train[:, 1], 0.0, atol=1e-12)
    assert np.allclose(scaled.X_val[:, 1], 3.0, atol=1e-12)
    assert np.allclose(scaled.X_test[:, 1], -3.0, atol=1e-12)

    # Shifted validation/test distributions must NOT be centered at 0
    assert abs(float(np.mean(scaled.X_val[:, 0]))) > 10.0
    assert abs(float(np.mean(scaled.X_test[:, 0]))) > 10.0


def test_split_input_validation() -> None:
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="ratios must sum to 1.0"):
        stratified_train_val_test_split(X, y, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
    with pytest.raises(ValueError, match="seeds must be distinct"):
        generate_preprocessed_splits(X, y, seeds=(42, 42))
