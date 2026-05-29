"""
dataset_generator.py  -  A03 Ensemble Learning  (seed=065)
Generates all datasets used in experiments.py.
Run standalone to verify shapes: python dataset_generator.py
"""
from __future__ import annotations
import numpy as np

SEED = 65  # last 3 digits of BSCS23065

def _rng():
    return np.random.default_rng(SEED)

# 1. LOW-NOISE
def make_low_noise(n=2000, n_informative=5, n_noise=5, n_classes=2):
    """
    Well-separated Gaussian clusters per class.
    Satisfied : clear class signal, balanced classes.
    Violated  : none major.
    Behavior  : all ensembles do well; single tree already near perfect.
    Bias-var  : low bias; ensemble reduces remaining variance.
    """
    rng = _rng()
    X_parts, y_parts = [], []
    per_class = n // n_classes
    for c in range(n_classes):
        center = rng.uniform(-5, 5, size=n_informative)
        Xi = rng.normal(center, 0.8, size=(per_class, n_informative))
        Xn = rng.normal(0, 1, size=(per_class, n_noise))
        X_parts.append(np.hstack([Xi, Xn]))
        y_parts.append(np.full(per_class, c))
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

# 2. HIGH-NOISE
def make_high_noise(n=2000, n_informative=5, n_noise=10, noise_rate=0.20):
    """
    Overlapping Gaussian + 20% label noise.
    Satisfied : enough samples for ensemble learning.
    Violated  : label noise; overlapping class distributions.
    Behavior  : bagging reduces variance; boosting overfits noisy samples.
    Bias-var  : high variance single tree; boosting raises bias on mislabeled.
    """
    rng = _rng()
    d = n_informative + n_noise
    X = rng.normal(0, 1, size=(n, d))
    signal = X[:, :n_informative].sum(axis=1)
    y = (signal > 0).astype(int)
    flip_idx = rng.choice(n, size=int(n * noise_rate), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y

# 3. HIGH-DIMENSIONAL  (n>=5000, d>=10)
def make_high_dim(n=5000, n_informative=10, n_noise=100):
    """
    n=5000, d=110. Many noisy features.
    Satisfied : RF feature subsampling directly addresses this setting.
    Violated  : many irrelevant features stress single-tree splits.
    Behavior  : RF >> Bagging > single tree due to forced feature diversity.
    Bias-var  : bagging still picks noisy features; RF subsampling fixes this.
    """
    rng = _rng()
    informative = rng.normal(0, 1, (n, n_informative))
    noise = rng.normal(0, 1, (n, n_noise))
    X = np.hstack([informative, noise])
    y = (informative[:, 0] + informative[:, 1] > 0).astype(int)
    flip_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y

# 4. IMBALANCED  (majority/minority >= 4)
def make_imbalanced(n=2000, imbalance_ratio=5):
    """
    Majority/minority ratio = 5 >= 4 as required.
    Satisfied : clear mean separation between classes.
    Violated  : class imbalance biases majority-vote ensembles.
    Behavior  : bagging/RF biased toward majority; AdaBoost reweights minority.
    Bias-var  : high bias toward majority without reweighting.
    """
    rng = _rng()
    n_minority = n // (imbalance_ratio + 1)
    n_majority = n - n_minority
    X_maj = rng.normal(np.zeros(10), 1, (n_majority, 10))
    X_min = rng.normal(np.ones(10)*3, 1, (n_minority, 10))
    Xn_maj = rng.normal(0, 1, (n_majority, 5))
    Xn_min = rng.normal(0, 1, (n_minority, 5))
    X = np.vstack([np.hstack([X_maj, Xn_maj]), np.hstack([X_min, Xn_min])])
    y = np.hstack([np.zeros(n_majority, int), np.ones(n_minority, int)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

# 5. BAGGING HELPS  - high variance due to nonlinear boundary + noise
def make_bagging_helps(n=3000):
    """
    Circular decision boundary + 15% label noise + 20 features.
    Single tree is highly unstable (high variance). Bagging helps significantly.
    Satisfied : nonlinear boundary exists; enough samples.
    Violated  : noisy labels; boundary not axis-aligned (hard for single tree).
    Behavior  : single tree accuracy ~0.70 with high variance; bagging ~0.79.
    Bias-var  : single tree: high variance; bagging: reduces variance by ~50%.
    """
    rng = _rng()
    X = rng.normal(0, 1, (n, 20))
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1.5).astype(int)
    noise_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    return X, y

# 6. BAGGING DOES NOT HELP  - low variance, correlated features
def make_bagging_nohelp(n=2000):
    """
    Very clean linear boundary. Features are highly correlated with feature 0.
    Bootstrap samples are nearly identical -> low diversity -> bagging adds nothing.
    Satisfied : clear linear signal.
    Violated  : correlated features make bootstrap samples nearly identical.
    Behavior  : single tree ~1.00; bagging ~1.00 (no gain).
    Bias-var  : single tree already has near-zero variance; nothing to reduce.
    """
    rng = _rng()
    X = rng.normal(0, 1, (n, 15))
    y = (X[:, 0] > 0).astype(int)
    for j in range(1, 10):
        X[:, j] = X[:, 0] + rng.normal(0, 0.05, n)
    return X, y

# 7. FEATURE DOMINANCE  (for RF Part B)
def make_feature_dominant(n=3000):
    """
    Feature 0 is a perfect predictor. Others are weak or noisy.
    Tests whether RF feature subsampling forces tree diversity.
    Satisfied : strong signal exists.
    Violated  : one feature dominates -> all trees identical without subsampling.
    Behavior  : Bagging trees all split on feature 0; RF forces use of other features.
    """
    rng = _rng()
    X = rng.normal(0, 1, (n, 20))
    y = (X[:, 0] > 0).astype(int)
    for j in range(1, 5):
        X[:, j] = X[:, 0] * 0.3 + rng.normal(0, 1, n)
    return X, y

# 8. BOOSTING NOISE datasets (for AdaBoost Part B)
def make_boosting_noise(n=1500, noise_rate=0.05):
    """
    Linear boundary with controllable label noise (5%, 15%, 30%).
    Satisfied : clear signal in informative features.
    Violated  : label noise - AdaBoost assigns high weights to noisy samples.
    Behavior  : AdaBoost overfocuses on mislabeled samples and overfits.
    """
    rng = _rng()
    X = rng.normal(0, 1, (n, 15))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    flip_idx = rng.choice(n, size=int(n * noise_rate), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y

# generate_all
def generate_all():
    datasets = {
        "low_noise":        make_low_noise(),
        "high_noise":       make_high_noise(),
        "high_dim":         make_high_dim(),
        "imbalanced":       make_imbalanced(),
        "bagging_helps":    make_bagging_helps(),
        "bagging_nohelp":   make_bagging_nohelp(),
        "feature_dominant": make_feature_dominant(),
        "boost_noise_05":   make_boosting_noise(noise_rate=0.05),
        "boost_noise_15":   make_boosting_noise(noise_rate=0.15),
        "boost_noise_30":   make_boosting_noise(noise_rate=0.30),
    }
    return datasets

if __name__ == "__main__":
    datasets = generate_all()
    for name, (X, y) in datasets.items():
        classes, counts = np.unique(y, return_counts=True)
        ratio = counts.max() / counts.min()
        print(f"{name:25s}  n={X.shape[0]:5d}  d={X.shape[1]:4d}  "
              f"counts={counts.tolist()}  imbalance={ratio:.1f}x")
