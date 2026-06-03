from __future__ import annotations

import numpy as np

from .c45_tree import DecisionTreeC45
from .utils import bootstrap_sample, accuracy


class RandomForestClassifier:
    """Random Forest from scratch: bootstrap + random feature subsets + majority vote."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        max_features: str | int = "sqrt",  # "sqrt", "log2", or int
        random_state: int = 65,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_: list[DecisionTreeC45] = []
        self.feature_subsets_: list[np.ndarray] = []
        self.oob_masks_: list[np.ndarray] = []
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: np.ndarray | None = None

    def _n_features_to_select(self, d: int) -> int:
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, d))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        raise ValueError(f"Unknown max_features: {self.max_features}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        self.n_classes_ = int(np.max(y) + 1)
        self.n_features_ = d
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.feature_subsets_ = []
        self.oob_masks_ = []
        feat_usage = np.zeros(d, dtype=int)
        k = self._n_features_to_select(d)

        for _ in range(self.n_estimators):
            # bootstrap
            X_b, y_b, oob_mask, _ = bootstrap_sample(X, y, rng)
            # random feature subset
            feat_idx = rng.choice(d, size=k, replace=False)
            feat_idx = np.sort(feat_idx)
            feat_usage[feat_idx] += 1

            tree = DecisionTreeC45(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion="gain_ratio",
                feature_indices=feat_idx,
            )
            tree.fit(X_b, y_b, n_classes=self.n_classes_)
            self.estimators_.append(tree)
            self.feature_subsets_.append(feat_idx)
            self.oob_masks_.append(oob_mask)

        self.feature_importances_ = feat_usage / feat_usage.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        votes = np.array([t.predict(X) for t in self.estimators_])
        out = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            counts = np.bincount(votes[:, i], minlength=self.n_classes_)
            out[i] = int(np.argmax(counts))
        return out

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """OOB accuracy: each sample evaluated only by trees that did not train on it."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = X.shape[0]
        oob_votes = np.zeros((n, self.n_classes_), dtype=int)

        for tree, oob_mask in zip(self.estimators_, self.oob_masks_):
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(X[oob_idx])
            for idx, pred in zip(oob_idx, preds):
                oob_votes[idx, pred] += 1

        voted = oob_votes.sum(axis=1) > 0
        if not np.any(voted):
            return float("nan")
        pred_oob = np.argmax(oob_votes[voted], axis=1)
        return accuracy(y[voted], pred_oob)

    def feature_usage_counts(self) -> np.ndarray:
        """How many times each feature was selected across all trees."""
        d = self.n_features_
        counts = np.zeros(d, dtype=int)
        for feat_idx in self.feature_subsets_:
            counts[feat_idx] += 1
        return counts
    def score(self, X, y):
        return float(sum(self.predict(X) == y) / len(y))
