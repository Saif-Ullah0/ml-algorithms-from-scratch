"""
experiments.py  -  A03 Ensemble Learning  (seed=065)
Run: python experiments.py --all
Outputs saved to report_assets/
"""
from __future__ import annotations
import argparse, sys, time, tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).resolve().parent
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from dataset_generator import generate_all, make_boosting_noise
from src.utils          import train_val_test_split, accuracy, DEFAULT_SEED
from src.bagging        import BaggingClassifier
from src.random_forest  import RandomForestClassifier
from src.adaboost       import AdaBoostClassifier
from src.c45_tree       import DecisionTreeC45

LOG: list[str] = []

def log(msg: str):
    print(msg); LOG.append(msg)

def save(name: str):
    p = ASSETS / name
    plt.savefig(str(p), dpi=110, bbox_inches="tight")
    plt.close()
    log(f"saved: {p}")

def split(X, y):
    return train_val_test_split(X, y, rng=np.random.default_rng(DEFAULT_SEED))

# ═══════════════════════════════════════════════════════════
# Q1  BAGGING
# ═══════════════════════════════════════════════════════════

def run_q1(datasets):
    log("\n=== Q1: Bagging ===")

    # ── Part A: Bootstrap visualization ──────────────────────
    X, y = datasets["low_noise"]
    rng  = np.random.default_rng(DEFAULT_SEED)
    n    = X.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, trial in zip(axes, range(3)):
        idx      = rng.integers(0, n, size=n)
        counts   = np.bincount(idx, minlength=n)
        oob_pct  = 100 * (counts == 0).mean()
        ax.bar(range(80), counts[:80], color="#4a90d9", alpha=0.75)
        ax.axhline(1, color="gray", linestyle="--", lw=0.8)
        ax.set_title(f"Bootstrap {trial+1}  |  OOB: {oob_pct:.1f}%")
        ax.set_xlabel("Sample index (first 80)")
        ax.set_ylabel("Times selected")
    fig.suptitle("Bootstrap Sampling - Repeated and Omitted Samples (seed=065)", fontsize=13)
    plt.tight_layout(); save("q1_bootstrap_visualization.png")

    # ── Part B: vary n_trees ──────────────────────────────────
    X, y  = datasets["bagging_helps"]
    Xtr, ytr, Xv, yv, Xte, yte = split(X, y)
    n_trees_list = [1, 5, 10, 20, 50]
    tr_accs, va_accs, variances = [], [], []

    for n_t in n_trees_list:
        bag = BaggingClassifier(n_estimators=n_t, max_depth=None, random_state=DEFAULT_SEED)
        bag.fit(Xtr, ytr)
        tr_accs.append(accuracy(ytr, bag.predict(Xtr)))
        va_accs.append(accuracy(yv,  bag.predict(Xv)))
        variances.append(bag.variance_across_trees(Xv))
        log(f"  n_trees={n_t:3d}: train={tr_accs[-1]:.4f}  val={va_accs[-1]:.4f}  var={variances[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(n_trees_list, tr_accs, "o-", label="Train", color="#2d6a4f")
    ax1.plot(n_trees_list, va_accs, "s-", label="Val",   color="#e07a5f")
    ax1.set_xlabel("Number of Trees"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Bagging: Accuracy vs Number of Trees")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(n_trees_list, variances, "^-", color="#6d6875")
    ax2.set_xlabel("Number of Trees"); ax2.set_ylabel("Prediction Variance")
    ax2.set_title("Bagging: Variance vs Number of Trees")
    ax2.grid(alpha=0.3)
    plt.tight_layout(); save("q1_bagging_ntrees.png")

    # vary depth
    depths       = [2, 4, 8, None]
    depth_labels = ["2", "4", "8", "None"]
    tr2, va2     = [], []
    for d in depths:
        bag = BaggingClassifier(n_estimators=30, max_depth=d, random_state=DEFAULT_SEED)
        bag.fit(Xtr, ytr)
        tr2.append(accuracy(ytr, bag.predict(Xtr)))
        va2.append(accuracy(yv,  bag.predict(Xv)))
    log(f"  depth sweep train: {[round(a,4) for a in tr2]}")
    log(f"  depth sweep val:   {[round(a,4) for a in va2]}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_labels, tr2, "o-", label="Train", color="#2d6a4f")
    ax.plot(depth_labels, va2, "s-", label="Val",   color="#e07a5f")
    ax.set_xlabel("Max Depth"); ax.set_ylabel("Accuracy")
    ax.set_title("Bagging: Accuracy vs Tree Depth")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save("q1_bagging_depth.png")

    # ── Part C: failure analysis ──────────────────────────────
    log("\n--- Q1 Part C: Failure Analysis ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (ds_name, label) in zip(axes, [
        ("bagging_helps",  "High-Variance (bagging helps)"),
        ("bagging_nohelp", "Correlated/Linear (bagging no help)"),
    ]):
        X, y = datasets[ds_name]
        Xtr, ytr, Xv, yv, _, _ = split(X, y)

        single = DecisionTreeC45(max_depth=None)
        single.fit(Xtr, ytr, n_classes=2)
        sv = accuracy(yv, single.predict(Xv))

        ns    = [1, 5, 10, 20, 50]
        vaccs = []
        for n_t in ns:
            bag = BaggingClassifier(n_estimators=n_t, max_depth=None, random_state=DEFAULT_SEED)
            bag.fit(Xtr, ytr)
            vaccs.append(accuracy(yv, bag.predict(Xv)))

        ax.plot(ns, vaccs, "o-", color="#4a90d9", label="Bagging")
        ax.axhline(sv, color="#e07a5f", linestyle="--", label=f"Single tree ({sv:.3f})")
        ax.set_xlabel("Number of Trees"); ax.set_ylabel("Val Accuracy")
        ax.set_title(label); ax.legend(); ax.grid(alpha=0.3)
        log(f"  {ds_name}: single={sv:.4f}  bag50={vaccs[-1]:.4f}  gain={vaccs[-1]-sv:+.4f}")

    plt.tight_layout(); save("q1_bagging_failure.png")


# ═══════════════════════════════════════════════════════════
# Q2  RANDOM FOREST
# ═══════════════════════════════════════════════════════════

def run_q2(datasets):
    log("\n=== Q2: Random Forest ===")

    # ── Part A+C: RF vs Bagging + OOB ────────────────────────
    X, y = datasets["low_noise"]
    Xtr, ytr, Xv, yv, Xte, yte = split(X, y)

    n_trees_list = [5, 10, 20, 50]
    rf_val, bag_val, rf_oob = [], [], []

    for n_t in n_trees_list:
        rf = RandomForestClassifier(n_estimators=n_t, max_depth=8, random_state=DEFAULT_SEED)
        rf.fit(Xtr, ytr)
        rf_val.append(accuracy(yv,  rf.predict(Xv)))
        rf_oob.append(rf.oob_score(Xtr, ytr))

        bag = BaggingClassifier(n_estimators=n_t, max_depth=8, random_state=DEFAULT_SEED)
        bag.fit(Xtr, ytr)
        bag_val.append(accuracy(yv, bag.predict(Xv)))
        log(f"  n_trees={n_t:3d}: RF_val={rf_val[-1]:.4f}  RF_oob={rf_oob[-1]:.4f}  Bag_val={bag_val[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(n_trees_list, rf_val,  "o-",  label="RF Val",     color="#4a90d9")
    ax1.plot(n_trees_list, bag_val, "s-",  label="Bagging Val",color="#e07a5f")
    ax1.plot(n_trees_list, rf_oob,  "^--", label="RF OOB",     color="#2d6a4f")
    ax1.set_xlabel("Number of Trees"); ax1.set_ylabel("Accuracy")
    ax1.set_title("RF vs Bagging + OOB Estimate")
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── Part B: feature dominance ─────────────────────────────
    Xfd, yfd = datasets["feature_dominant"]
    Xtr2, ytr2, Xv2, yv2, _, _ = split(Xfd, yfd)

    rf_fd = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=DEFAULT_SEED)
    rf_fd.fit(Xtr2, ytr2)
    usage = rf_fd.feature_usage_counts()

    ax2.bar(range(len(usage)), usage, color="#6d6875", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5, label="Dominant feature (0)")
    ax2.set_xlabel("Feature Index"); ax2.set_ylabel("Times Selected")
    ax2.set_title("RF Feature Usage - Dominant Feature Dataset")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout(); save("q2_rf_oob_features.png")

    fd_val = accuracy(yv2,  rf_fd.predict(Xv2))
    fd_oob = rf_fd.oob_score(Xtr2, ytr2)
    log(f"  feature_dominant: val={fd_val:.4f}  oob={fd_oob:.4f}")
    log(f"  top-3 used features: {np.argsort(usage)[::-1][:3].tolist()}")

    # ── Part D: high-dimensional ──────────────────────────────
    log("\n--- Q2 Part D: High-Dimensional ---")
    X, y = datasets["high_dim"]
    Xtr, ytr, Xv, yv, _, _ = split(X, y)

    configs = [
        ("Single Tree",  DecisionTreeC45,       {"max_depth": 8}),
        ("Bagging(20)",  BaggingClassifier,      {"n_estimators": 20, "max_depth": 8, "random_state": DEFAULT_SEED}),
        ("RF(20,sqrt)",  RandomForestClassifier, {"n_estimators": 20, "max_depth": 8, "max_features": "sqrt", "random_state": DEFAULT_SEED}),
    ]
    res = {}
    for label, Model, kwargs in configs:
        tracemalloc.start()
        t0 = time.time()
        if label == "Single Tree":
            m = Model(**kwargs); m.fit(Xtr, ytr, n_classes=2)
        else:
            m = Model(**kwargs); m.fit(Xtr, ytr)
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        tr_a = accuracy(ytr, m.predict(Xtr))
        va_a = accuracy(yv,  m.predict(Xv))
        res[label] = (tr_a, va_a, elapsed, peak/1e6)
        log(f"  {label:15s}: train={tr_a:.4f}  val={va_a:.4f}  time={elapsed:.1f}s  mem={peak/1e6:.1f}MB")

    labels  = list(res.keys())
    tr_vals = [res[k][0] for k in labels]
    va_vals = [res[k][1] for k in labels]
    times   = [res[k][2] for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels)); w = 0.35
    ax1.bar(x-w/2, tr_vals, w, label="Train", color="#4a90d9", alpha=0.85)
    ax1.bar(x+w/2, va_vals, w, label="Val",   color="#e07a5f", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10)
    ax1.set_ylabel("Accuracy"); ax1.set_title("High-Dim: Train vs Val Accuracy")
    ax1.legend(); ax1.set_ylim(0.4, 1.05); ax1.grid(alpha=0.3, axis="y")

    ax2.bar(labels, times, color="#6d6875", alpha=0.85)
    ax2.set_ylabel("Training Time (s)"); ax2.set_title("High-Dim: Training Time")
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout(); save("q2_high_dim.png")


# ═══════════════════════════════════════════════════════════
# Q3  ADABOOST
# ═══════════════════════════════════════════════════════════

def run_q3(datasets):
    log("\n=== Q3: AdaBoost ===")

    # ── Part A: basic AdaBoost ────────────────────────────────
    X, y = datasets["low_noise"]
    Xtr, ytr, Xv, yv, _, _ = split(X, y)

    ada = AdaBoostClassifier(n_estimators=50, stump_depth=1, random_state=DEFAULT_SEED)
    ada.fit(Xtr, ytr)
    val_accs = [accuracy(yv, p) for p in ada.staged_predict(Xv)]
    log(f"  AdaBoost basic: train={ada.train_acc_history_[-1]:.4f}  val={val_accs[-1]:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(ada.train_acc_history_, label="Train", color="#2d6a4f")
    plt.plot(val_accs,               label="Val",   color="#e07a5f")
    plt.xlabel("Boosting Round"); plt.ylabel("Accuracy")
    plt.title("AdaBoost: Train vs Val (low_noise, depth=1 stumps)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); save("q3_adaboost_basic.png")

    # ── Part B: noise sensitivity ─────────────────────────────
    log("\n--- Q3 Part B: Noise Sensitivity ---")
    noise_rates = [0.05, 0.15, 0.30]
    colors      = ["#2d6a4f", "#e07a5f", "#6d6875"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for rate, color in zip(noise_rates, colors):
        X, y = make_boosting_noise(n=1500, noise_rate=rate)
        Xtr, ytr, Xv, yv, _, _ = split(X, y)
        ada = AdaBoostClassifier(n_estimators=50, stump_depth=1, random_state=DEFAULT_SEED)
        ada.fit(Xtr, ytr)
        val_accs = [accuracy(yv, p) for p in ada.staged_predict(Xv)]
        axes[0].plot(ada.train_acc_history_, color=color, label=f"noise={int(rate*100)}%")
        axes[1].plot(val_accs,               color=color, label=f"noise={int(rate*100)}%")
        log(f"  noise={int(rate*100):2d}%: train={ada.train_acc_history_[-1]:.4f}  val={val_accs[-1]:.4f}")
        if len(ada.weight_history_) > 10:
            w = ada.weight_history_[10]
            log(f"    weights@r10: max={w.max():.5f}  top1pct={w[w>np.percentile(w,99)].sum():.4f}")

    for ax, title in zip(axes, ["Train Accuracy", "Val Accuracy"]):
        ax.set_xlabel("Boosting Round"); ax.set_ylabel("Accuracy")
        ax.set_title(f"AdaBoost Noise Sensitivity - {title}")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save("q3_noise_sensitivity.png")

    # weight distribution evolution at 30% noise
    X, y = make_boosting_noise(n=1500, noise_rate=0.30)
    Xtr, ytr, _, _, _, _ = split(X, y)
    ada30 = AdaBoostClassifier(n_estimators=50, stump_depth=1, random_state=DEFAULT_SEED)
    ada30.fit(Xtr, ytr)

    rounds = [0, 10, 25, 49]
    fig, axes = plt.subplots(1, len(rounds), figsize=(16, 3))
    for ax, r in zip(axes, rounds):
        if r < len(ada30.weight_history_):
            ax.hist(ada30.weight_history_[r], bins=40, color="#6d6875", alpha=0.8)
            ax.set_title(f"Round {r}"); ax.set_xlabel("Weight"); ax.set_ylabel("Count")
    fig.suptitle("Weight Distribution Evolution (30% noise)", fontsize=12)
    plt.tight_layout(); save("q3_weight_evolution.png")

    # ── Part C: weak vs strong learner ───────────────────────
    log("\n--- Q3 Part C: Weak vs Strong Learner ---")
    X, y = datasets["high_noise"]
    Xtr, ytr, Xv, yv, _, _ = split(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for depth, color, label in [(1, "#4a90d9", "Stumps (depth=1)"),
                                 (4, "#e07a5f", "Deep trees (depth=4)")]:
        ada = AdaBoostClassifier(n_estimators=50, stump_depth=depth, random_state=DEFAULT_SEED)
        ada.fit(Xtr, ytr)
        val_accs = [accuracy(yv, p) for p in ada.staged_predict(Xv)]
        axes[0].plot(ada.train_acc_history_, color=color, label=label)
        axes[1].plot(val_accs,               color=color, label=label)
        log(f"  depth={depth}: train={ada.train_acc_history_[-1]:.4f}  val={val_accs[-1]:.4f}")

    for ax, title in zip(axes, ["Train Accuracy", "Val Accuracy"]):
        ax.set_xlabel("Boosting Round"); ax.set_ylabel("Accuracy")
        ax.set_title(f"Weak vs Strong - {title}")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save("q3_weak_vs_strong.png")

    # ── Part D: final comparison ──────────────────────────────
    log("\n--- Q3 Part D: Final Comparison ---")
    X, y = datasets["low_noise"]
    Xtr, ytr, Xv, yv, Xte, yte = split(X, y)

    res = {}
    single = DecisionTreeC45(max_depth=8)
    single.fit(Xtr, ytr, n_classes=2)
    res["Single Tree"] = (accuracy(ytr, single.predict(Xtr)),
                          accuracy(yv,  single.predict(Xv)),
                          accuracy(yte, single.predict(Xte)))

    bag = BaggingClassifier(n_estimators=30, max_depth=8, random_state=DEFAULT_SEED)
    bag.fit(Xtr, ytr)
    res["Bagging"] = (accuracy(ytr, bag.predict(Xtr)),
                      accuracy(yv,  bag.predict(Xv)),
                      accuracy(yte, bag.predict(Xte)))

    rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=DEFAULT_SEED)
    rf.fit(Xtr, ytr)
    res["Random Forest"] = (accuracy(ytr, rf.predict(Xtr)),
                             accuracy(yv,  rf.predict(Xv)),
                             accuracy(yte, rf.predict(Xte)))

    ada = AdaBoostClassifier(n_estimators=50, stump_depth=1, random_state=DEFAULT_SEED)
    ada.fit(Xtr, ytr)
    res["AdaBoost"] = (accuracy(ytr, ada.predict(Xtr)),
                       accuracy(yv,  ada.predict(Xv)),
                       accuracy(yte, ada.predict(Xte)))

    for name, (tr, va, te) in res.items():
        log(f"  {name:15s}: train={tr:.4f}  val={va:.4f}  test={te:.4f}")

    labels  = list(res.keys())
    tr_a    = [res[k][0] for k in labels]
    va_a    = [res[k][1] for k in labels]
    te_a    = [res[k][2] for k in labels]

    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-w,   tr_a, w, label="Train", color="#4a90d9", alpha=0.85)
    ax.bar(x,     va_a, w, label="Val",   color="#e07a5f", alpha=0.85)
    ax.bar(x+w,   te_a, w, label="Test",  color="#2d6a4f", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy"); ax.set_title("Final Comparison: All Ensemble Methods")
    ax.legend(); ax.set_ylim(0.5, 1.05); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); save("q3_final_comparison.png")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--q1",  action="store_true")
    parser.add_argument("--q2",  action="store_true")
    parser.add_argument("--q3",  action="store_true")
    args = parser.parse_args()
    run_all = args.all or not any([args.q1, args.q2, args.q3])

    log(f"seed={DEFAULT_SEED}")
    log("Loading datasets...")
    datasets = generate_all()
    for name, (X, y) in datasets.items():
        log(f"  {name:25s}  n={X.shape[0]}  d={X.shape[1]}")

    if run_all or args.q1: run_q1(datasets)
    if run_all or args.q2: run_q2(datasets)
    if run_all or args.q3: run_q3(datasets)

    log_path = ASSETS / "run_log.txt"
    log_path.write_text("\n".join(LOG), encoding="utf-8")
    log(f"saved: {log_path}")

if __name__ == "__main__":
    main()
