from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import scipy.io

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

_CREDITCARD_FEATURES = [*(f"V{i}" for i in range(1, 29)), "Amount"]


def _resolve_dataset_path(
    filename: str, data_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve a dataset path under data/raw (or data_dir) and validate existence."""
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_RAW_DIR
    file_path = base_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset file '{filename}' at '{file_path}'. "
            f"Please ensure '{filename}' is placed in '{base_dir}' (expected under data/raw/)."
        )
    return file_path


def _validate_loaded_arrays(
    X: np.ndarray, y: np.ndarray, dataset_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Cast features to float64 and binary labels to 1D int64, validating shapes."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).ravel()

    if X_arr.ndim != 2:
        raise ValueError(f"{dataset_name}: expected 2D feature matrix, got shape {X_arr.shape}.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"{dataset_name}: sample count mismatch between X ({X_arr.shape[0]}) and y ({y_arr.shape[0]})."
        )
    unique_labels = set(np.unique(y_arr).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: expected binary labels in {{0, 1}}, found {sorted(unique_labels)}."
        )
    return X_arr, y_arr


def load_thyroid(
    data_dir: Optional[Union[str, Path]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the ODDS Thyroid dataset (n=3,772, d=6) from thyroid.mat."""
    path = _resolve_dataset_path("thyroid.mat", data_dir)
    mat_data = scipy.io.loadmat(path)
    if "X" not in mat_data or "y" not in mat_data:
        raise KeyError(f"Expected 'X' and 'y' keys in '{path}', found {list(mat_data.keys())}.")
    return _validate_loaded_arrays(mat_data["X"], mat_data["y"], "thyroid")


def load_cardio(
    data_dir: Optional[Union[str, Path]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the ADBench Cardio dataset (n=1,831, d=21) from 6_cardio.npz."""
    path = _resolve_dataset_path("6_cardio.npz", data_dir)
    with np.load(path) as npz_data:
        if "X" not in npz_data or "y" not in npz_data:
            raise KeyError(f"Expected 'X' and 'y' keys in '{path}', found {list(npz_data.files)}.")
        X = npz_data["X"]
        y = npz_data["y"]
    return _validate_loaded_arrays(X, y, "cardio")


def load_creditcard(
    data_dir: Optional[Union[str, Path]] = None,
    n_samples: int = 20_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Kaggle Credit Card Fraud dataset from creditcard.csv.

    Drops 'Time', retains V1-V28 and 'Amount' (d=29), keeps all fraud rows (y=1),
    samples legitimate rows (y=0) up to n_samples, and deterministically shuffles
    the resulting dataset using random_state.
    """
    path = _resolve_dataset_path("creditcard.csv", data_dir)
    df = pd.read_csv(path)

    expected_cols = set(_CREDITCARD_FEATURES) | {"Class"}
    missing_cols = sorted(expected_cols - set(df.columns))
    if missing_cols:
        raise KeyError(f"Missing expected columns in '{path}': {missing_cols}.")

    X_full = df[_CREDITCARD_FEATURES].to_numpy(dtype=np.float64)
    y_full = df["Class"].to_numpy(dtype=np.int64)

    fraud_indices = np.flatnonzero(y_full == 1)
    normal_indices = np.flatnonzero(y_full == 0)

    n_fraud = len(fraud_indices)
    if n_samples < n_fraud:
        raise ValueError(
            f"n_samples ({n_samples}) cannot be smaller than total fraud cases ({n_fraud})."
        )
    if n_samples > len(y_full):
        raise ValueError(
            f"n_samples ({n_samples}) exceeds dataset size ({len(y_full)})."
        )

    n_normal_to_sample = n_samples - n_fraud
    rng = np.random.default_rng(random_state)
    sampled_normal = rng.choice(normal_indices, size=n_normal_to_sample, replace=False)

    combined_indices = np.concatenate([fraud_indices, sampled_normal])
    shuffled_indices = combined_indices[rng.permutation(len(combined_indices))]

    return _validate_loaded_arrays(
        X_full[shuffled_indices], y_full[shuffled_indices], "creditcard"
    )


DATASET_LOADERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "thyroid": load_thyroid,
    "cardio": load_cardio,
    "creditcard": load_creditcard,
}


def load_dataset(
    name: str, data_dir: Optional[Union[str, Path]] = None, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Load a supported anomaly detection dataset by name ('thyroid', 'cardio', 'creditcard')."""
    key = name.strip().lower()
    if key not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported datasets: {sorted(DATASET_LOADERS.keys())}."
        )
    return DATASET_LOADERS[key](data_dir=data_dir, **kwargs)
