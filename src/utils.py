from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_SEED = 65  # roll last-3 digits: 065


def get_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
):
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("splits must sum to 1")
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = rng or get_rng()
        rng.shuffle(idx)
    n_train = int(round(n * train_size))
    n_val = int(round(n * val_size))
    i_train = idx[:n_train]
    i_val = idx[n_train: n_train + n_val]
    i_test = idx[n_train + n_val:]

    def take(arr, ind):
        return None if arr is None else arr[ind]

    return (X[i_train], take(y, i_train),
            X[i_val],   take(y, i_val),
            X[i_test],  take(y, i_test))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def bootstrap_sample(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Draw bootstrap sample (with replacement). Returns (X_boot, y_boot, oob_mask)."""
    n = X.shape[0]
    indices = rng.integers(0, n, size=n)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[indices] = False
    return X[indices], y[indices], oob_mask, indices
