from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import entropy_from_counts


@dataclass
class Node:
    is_leaf: bool
    pred_class: int
    class_counts: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


def _entropy_from_y(y: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, minlength=n_classes)
    return entropy_from_counts(counts)


class DecisionTreeC45:
    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_gain_ratio: float = 0.0,
        criterion: str = "gain_ratio",
        use_c45_ig_filter: bool = True,
        feature_indices: np.ndarray | None = None,  # for Random Forest feature subsampling
    ):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_gain_ratio = float(min_gain_ratio)
        if criterion not in {"gain_ratio", "info_gain"}:
            raise ValueError("criterion must be 'gain_ratio' or 'info_gain'")
        self.criterion = criterion
        self.use_c45_ig_filter = bool(use_c45_ig_filter)
        self.feature_indices = feature_indices  # subset of features to consider at each split

        self.n_classes_: int | None = None
        self.root_: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, n_classes: int | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = int(n_classes) if n_classes is not None else int(np.max(y) + 1)
        self.root_ = self._build(X, y, depth=0)
        return self

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        k = self.n_classes_
        parent_entropy = _entropy_from_y(y, k)

        # feature subset for Random Forest
        feat_range = self.feature_indices if self.feature_indices is not None else np.arange(d)

        Y_oh = np.eye(k, dtype=int)[y]

        gr_candidates = []
        ig_candidates = []

        def ent_from_counts(counts):
            counts = counts.astype(float)
            s = counts.sum(axis=1, keepdims=True)
            p = np.divide(counts, s, out=np.zeros_like(counts), where=s > 0)
            logp = np.log2(np.clip(p, 1e-12, 1.0))
            return -np.sum(p * logp, axis=1)

        for j in feat_range:
            xj = X[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs = xj[order]
            Ys = Y_oh[order]

            diff = xs[1:] != xs[:-1]
            if not np.any(diff):
                continue

            cum = np.cumsum(Ys, axis=0)
            total = cum[-1]

            idx = np.where(diff)[0]
            left_counts = cum[idx]
            right_counts = total - left_counts

            n_left = left_counts.sum(axis=1).astype(float)
            n_right = right_counts.sum(axis=1).astype(float)

            h_left = ent_from_counts(left_counts)
            h_right = ent_from_counts(right_counts)

            info_gain = parent_entropy - (n_left / n) * h_left - (n_right / n) * h_right

            p_left = n_left / n
            p_right = n_right / n
            split_info = -(
                p_left * np.log2(np.where(p_left > 0, p_left, 1.0))
                + p_right * np.log2(np.where(p_right > 0, p_right, 1.0))
            )

            gain_ratio = np.divide(info_gain, split_info, out=np.zeros_like(info_gain), where=split_info > 1e-12)

            i_gr = int(np.argmax(gain_ratio))
            split_i = int(idx[i_gr])
            thr_gr = float((xs[split_i] + xs[split_i + 1]) / 2.0)
            gr_candidates.append((j, thr_gr, float(gain_ratio[i_gr]), float(info_gain[i_gr]), float(split_info[i_gr])))

            i_ig = int(np.argmax(info_gain))
            split_i = int(idx[i_ig])
            thr_ig = float((xs[split_i] + xs[split_i + 1]) / 2.0)
            ig_candidates.append((j, thr_ig, float(gain_ratio[i_ig]), float(info_gain[i_ig]), float(split_info[i_ig])))

        if not ig_candidates:
            return -np.inf, (None, None, None, None, None), parent_entropy

        if self.criterion == "info_gain":
            best = max(ig_candidates, key=lambda t: t[3])
            return float(best[3]), best, parent_entropy

        if not self.use_c45_ig_filter:
            best = max(gr_candidates, key=lambda t: t[2])
            return float(best[2]), best, parent_entropy

        avg_ig = float(np.mean([t[3] for t in gr_candidates])) if gr_candidates else 0.0
        filtered = [t for t in gr_candidates if t[3] >= avg_ig]
        pool = filtered if filtered else gr_candidates
        best = max(pool, key=lambda t: t[2])
        return float(best[2]), best, parent_entropy

    def _build(self, X: np.ndarray, y: np.ndarray, *, depth: int) -> Node:
        k = self.n_classes_
        counts = np.bincount(y, minlength=k)
        pred = int(np.argmax(counts))

        if np.count_nonzero(counts) == 1:
            return Node(True, pred, counts)
        if X.shape[0] < self.min_samples_split:
            return Node(True, pred, counts)
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(True, pred, counts)

        best_score, best, _ = self._best_split(X, y)
        feat, thr, gain_ratio, info_gain, split_info = best
        if feat is None or thr is None:
            return Node(True, pred, counts)

        if float(gain_ratio) < self.min_gain_ratio:
            return Node(True, pred, counts)

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return Node(True, pred, counts)

        left = self._build(X[left_mask], y[left_mask], depth=depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth=depth + 1)
        return Node(False, pred, counts, feature_index=int(feat), threshold=float(thr), left=left, right=right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        out = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.pred_class
        return out

    def n_nodes(self) -> int:
        if self.root_ is None:
            return 0
        stack = [self.root_]
        c = 0
        while stack:
            n = stack.pop()
            c += 1
            if not n.is_leaf:
                stack.append(n.left)
                stack.append(n.right)
        return c

    def root_split(self):
        if self.root_ is None or self.root_.is_leaf:
            return None
        return (self.root_.feature_index, self.root_.threshold)

    def to_text(self, feature_names=None, *, max_depth=5):
        if self.root_ is None:
            return "(unfitted)"

        def fname(i):
            return f"x[{i}]" if feature_names is None else feature_names[i]

        lines = []

        def rec(node, depth, prefix):
            if depth > max_depth:
                lines.append(prefix + "...")
                return
            if node.is_leaf:
                lines.append(prefix + f"Leaf -> class {node.pred_class} counts={node.class_counts.tolist()}")
                return
            lines.append(prefix + f"if {fname(node.feature_index)} <= {node.threshold:.4f}:")
            rec(node.left, depth + 1, prefix + "  ")
            lines.append(prefix + "else:")
            rec(node.right, depth + 1, prefix + "  ")

        rec(self.root_, 0, "")
        return "\n".join(lines)
