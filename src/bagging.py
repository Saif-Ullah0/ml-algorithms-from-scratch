from __future__ import annotations

import numpy as np

from .c45_tree import DecisionTreeC45
from .utils import bootstrap_sample, accuracy


class BaggingClassifier:
    """Bagging ensemble of DecisionTreeC45 trees, built from scratch."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        random_state: int = 65,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.estimators_: list[DecisionTreeC45] = []
        self.oob_indices_: list[np.ndarray] = []   # oob mask per tree
        self.n_classes_: int = 0
        self.train_indices_: list[np.ndarray] = [] # bootstrap indices per tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = int(np.max(y) + 1)
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.oob_indices_ = []
        self.train_indices_ = []

        for _ in range(self.n_estimators):
            X_b, y_b, oob_mask, boot_idx = bootstrap_sample(X, y, rng)
            tree = DecisionTreeC45(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion="gain_ratio",
            )
            tree.fit(X_b, y_b, n_classes=self.n_classes_)
            self.estimators_.append(tree)
            self.oob_indices_.append(oob_mask)
            self.train_indices_.append(boot_idx)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # collect votes: shape (n_estimators, n_samples)
        votes = np.array([t.predict(X) for t in self.estimators_])
        # majority vote
        out = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            counts = np.bincount(votes[:, i], minlength=self.n_classes_)
            out[i] = int(np.argmax(counts))
        return out

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute OOB accuracy manually."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = X.shape[0]
        oob_votes = np.zeros((n, self.n_classes_), dtype=int)

        for tree, oob_mask in zip(self.estimators_, self.oob_indices_):
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(X[oob_idx])
            for idx, pred in zip(oob_idx, preds):
                oob_votes[idx, pred] += 1

        # only score samples that appeared as OOB at least once
        voted = oob_votes.sum(axis=1) > 0
        if not np.any(voted):
            return float("nan")
        pred_oob = np.argmax(oob_votes[voted], axis=1)
        return accuracy(y[voted], pred_oob)

    def variance_across_trees(self, X: np.ndarray) -> float:
        """Mean prediction variance across trees (disagreement measure)."""
        X = np.asarray(X, dtype=float)
        votes = np.array([t.predict(X) for t in self.estimators_])  # (n_est, n)
        # fraction of trees agreeing with majority per sample
        majority = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=self.n_classes_).max(),
            axis=0, arr=votes
        )
        agreement = majority / self.n_estimators
        return float(1.0 - agreement.mean())  # higher = more disagreement = more variance
    def score(self, X, y):
        return float(sum(self.predict(X) == y) / len(y))
