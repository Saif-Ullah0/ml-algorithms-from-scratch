from __future__ import annotations

import numpy as np

from .c45_tree import DecisionTreeC45
from .utils import accuracy


class AdaBoostClassifier:
    """AdaBoost from scratch using decision stumps (depth=1 trees).

    Supports binary classification only (labels must be 0/1).
    Converted internally to +1/-1 for the algorithm.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        stump_depth: int = 1,   # 1 = decision stump; can increase for weak learner comparison
        random_state: int = 65,
    ):
        self.n_estimators = n_estimators
        self.stump_depth = stump_depth
        self.random_state = random_state

        self.estimators_: list[DecisionTreeC45] = []
        self.alphas_: list[float] = []
        self.weight_history_: list[np.ndarray] = []  # weights at each round
        self.train_acc_history_: list[float] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = X.shape[0]
        self.classes_ = np.unique(y)

        # convert to +1/-1 internally
        y_pm = np.where(y == self.classes_[1], 1, -1).astype(float)

        # uniform initial weights
        w = np.ones(n) / n

        self.estimators_ = []
        self.alphas_ = []
        self.weight_history_ = []
        self.train_acc_history_ = []

        for _ in range(self.n_estimators):
            self.weight_history_.append(w.copy())

            # sample indices according to weights (weighted bootstrap)
            rng = np.random.default_rng(self.random_state + len(self.estimators_))
            sample_idx = rng.choice(n, size=n, replace=True, p=w / w.sum())
            X_s = X[sample_idx]
            y_s = y[sample_idx]

            # train weak learner (stump)
            stump = DecisionTreeC45(
                max_depth=self.stump_depth,
                min_samples_split=2,
                criterion="info_gain",
            )
            stump.fit(X_s, y_s, n_classes=2)
            self.estimators_.append(stump)

            # predict on full training set
            pred = stump.predict(X)
            pred_pm = np.where(pred == self.classes_[1], 1, -1).astype(float)

            # weighted error
            incorrect = (pred_pm != y_pm).astype(float)
            eps = float(np.dot(w, incorrect) / w.sum())
            eps = np.clip(eps, 1e-10, 1 - 1e-10)

            # alpha (learner weight)
            alpha = 0.5 * np.log((1.0 - eps) / eps)
            self.alphas_.append(alpha)

            # update weights
            w = w * np.exp(-alpha * y_pm * pred_pm)
            w = w / w.sum()  # normalize

            # track training accuracy of ensemble so far
            ensemble_pred = self._predict_from(X, len(self.estimators_))
            self.train_acc_history_.append(accuracy(y, ensemble_pred))

        return self

    def _predict_from(self, X: np.ndarray, n_est: int) -> np.ndarray:
        """Predict using first n_est estimators."""
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for tree, alpha in zip(self.estimators_[:n_est], self.alphas_[:n_est]):
            pred = tree.predict(X)
            pred_pm = np.where(pred == self.classes_[1], 1, -1).astype(float)
            scores += alpha * pred_pm
        return np.where(scores >= 0, self.classes_[1], self.classes_[0]).astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_from(X, len(self.estimators_))

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after each boosting round."""
        for i in range(1, len(self.estimators_) + 1):
            yield self._predict_from(X, i)
