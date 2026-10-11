"""
Regionwise / Setwise BMA — Core Experiments
Verifies: C1 (closed-form weights), C2 (gamma-tau diagnostic),
          C3 (minority collapse + gate), C4 (subset selection), C5 (PAC-Bayes tightness)
"""
import numpy as np
from scipy.linalg import inv
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import os, json

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)
OUT_DIR = r"C:\Users\robot\Desktop\M PAPER\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Core: Closed-form BMA weights (Theorem 3.1)
# ============================================================
def min_variance_weights(Sigma, eps_min=None):
    """
    Theorem 3.1: w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
    If eps_min is provided (2-model case), enforce w_L >= eps_min (Theorem 6.2).
    """
    M = Sigma.shape[0]
    # Add small regularization for numerical stability
    Sigma_reg = Sigma + 1e-8 * np.eye(M)
    Sigma_inv = inv(Sigma_reg)
    one = np.ones(M)
    w = Sigma_inv @ one
    w = w / np.sum(w)
    # Clip to non-negative
    w = np.clip(w, 0, 1)
    w = w / np.sum(w)

    # Gate: enforce w_L >= eps_min (2-model case)
    if eps_min is not None and M == 2:
        if w[1] < eps_min:
            w = np.array([1.0 - eps_min, eps_min])
    return w


def compute_error_covariance(residuals):
    """
    Compute error covariance matrix from model residuals.
    residuals: (n_samples, M_models)
    Returns: Sigma (M x M)
    """
    M = residuals.shape[1]
    Sigma = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            Sigma[i, j] = np.mean(residuals[:, i] * residuals[:, j])
    return Sigma


# ============================================================
# Model Pool Factory
# ============================================================
def get_model_pool(task="classification"):
    if task == "classification":
        return [
            ("LogReg", LogisticRegression(max_iter=1000, random_state=SEED)),
            ("Ridge", Ridge(alpha=1.0, random_state=SEED)),
            ("RF", RandomForestClassifier(n_estimators=50, random_state=SEED)),
            ("GB", GradientBoostingClassifier(n_estimators=30, random_state=SEED)),
            ("SVM", SVC(probability=True, random_state=SEED)),
            ("kNN", KNeighborsClassifier(n_neighbors=5)),
        ]
    else:
        return [
            ("Ridge", Ridge(alpha=1.0, random_state=SEED)),
            ("Lasso", Lasso(alpha=0.1, random_state=SEED)),
            ("RF", RandomForestRegressor(n_estimators=50, random_state=SEED)),
            ("GB", GradientBoostingRegressor(n_estimators=30, random_state=SEED)),
        ]


# ============================================================
# Synthetic Data Generators
# ============================================================
def synth_two_region(n=2000, d=10, seed=SEED):
    """
    Synth-1: Two-region Gaussian.
    Region 1 (x1 > 0): model A (linear) is best.
    Region 2 (x1 <= 0): model B (nonlinear) is best.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    # True function: piecewise linear with different slopes
    y = np.zeros(n)
    mask1 = X[:, 0] > 0  # Region 1
    mask2 = ~mask1        # Region 2
    y[mask1] = X[mask1, 0] * 2 + X[mask1, 1] * 1 + rng.randn(mask1.sum()) * 0.5
    y[mask2] = X[mask2, 0] * (-1) + X[mask2, 2] * 1.5 + rng.randn(mask2.sum()) * 0.5
    # Classify for classification task
    y_cls = (y > np.median(y)).astype(int)
    return X, y, y_cls, mask1, mask2


def synth_minority_collapse(n_majority=1800, n_minority_list=[200, 100, 50, 20, 10],
                            d=5, seed=SEED):
    """
    Synth-3: Minority collapse experiment.
    Major region: global model works well.
    Minor region: local model needed, but few samples.
    Returns data for each imbalance ratio.
    """
    rng = np.random.RandomState(seed)
    results = {}
    for n_min in n_minority_list:
        # Majority region
        X_maj = rng.randn(n_majority, d)
        y_maj = X_maj[:, 0] * 1.5 + X_maj[:, 1] * 0.8 + rng.randn(n_majority) * 0.5
        # Minority region (different true function)
        X_min = rng.randn(n_min, d)
        y_min = X_min[:, 2] * 2.0 - X_min[:, 3] * 1.0 + rng.randn(n_min) * 0.3
        X = np.vstack([X_maj, X_min])
        y = np.concatenate([y_maj, y_min])
        mask_min = np.zeros(len(X), dtype=bool)
        mask_min[n_majority:] = True
        results[n_min] = (X, y, mask_min, n_majority)
    return results


def synth_m_model(n=1000, d=8, M=20, seed=SEED):
    """
    Synth-4: High-dimensional M-model experiment.
    Only 3-5 models are truly useful; rest are noise.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + rng.randn(n) * 0.5
    return X, y


# ============================================================
# Experiment 1: Two-Region Closed-Form Weights (C1)
# ============================================================
def exp1_closed_form_weights():
    print("=" * 60)
    print("Exp 1: Closed-Form Regionwise Weights (C1)")
    print("=" * 60)

    X, y_reg, y_cls, mask1, mask2 = synth_two_region(n=2000, d=10)
    X_train, X_test, y_train, y_test, m1_tr, m1_te = train_test_split(
        X, y_reg, mask1, test_size=0.3, random_state=SEED
    )

    # Train model pool (regression models)
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    models = [
        ("Ridge1", Ridge(alpha=0.1, random_state=SEED)),
        ("Ridge2", Ridge(alpha=1.0, random_state=SEED)),
        ("Ridge3", Ridge(alpha=10.0, random_state=SEED)),
        ("RF", RandomForestRegressor(n_estimators=50, random_state=SEED)),
        ("GB", GradientBoostingRegressor(n_estimators=30, random_state=SEED)),
    ]

    trained = {}
    for name, mdl in models:
        mdl.fit(X_train, y_train)
        trained[name] = mdl

    names = [m[0] for m in models]
    M = len(names)

    # Get predictions on train (for covariance estimation)
    preds_train = np.column_stack([trained[n].predict(X_train) for n in names])
    preds_test = np.column_stack([trained[n].predict(X_test) for n in names])

    # Global BMA (K=1): one weight vector for all
    residuals_global = preds_train - y_train[:, None]
    Sigma_global = compute_error_covariance(residuals_global)
    w_global = min_variance_weights(Sigma_global)
    pred_global = preds_test @ w_global
    mse_global = np.mean((pred_global - y_test) ** 2)

    # Regionwise BMA (K=2): separate weights per region
    results = {}
    for region_name, mask in [("Region1", m1_tr), ("Region2", ~m1_tr)]:
        residuals_r = residuals_global[mask]
        Sigma_r = compute_error_covariance(residuals_r)
        w_r = min_variance_weights(Sigma_r)
        results[region_name] = (Sigma_r, w_r)

    # Apply regionwise weights on test
    pred_rw = np.zeros(len(y_test))
    for region_name, mask_te in [("Region1", m1_te), ("Region2", ~m1_te)]:
        w_r = results[region_name][1]
        pred_rw[mask_te] = preds_test[mask_te] @ w_r
    mse_rw = np.mean((pred_rw - y_test) ** 2)

    # Single best model
    mse_each = [np.mean((preds_test[:, i] - y_test) ** 2) for i in range(M)]
    mse_single_best = min(mse_each)

    # Oracle (best per region)
    mse_oracle = 0
    for mask_te in [m1_te, ~m1_te]:
        mse_each_r = [np.mean((preds_test[mask_te, i] - y_test[mask_te]) ** 2) for i in range(M)]
        mse_oracle += min(mse_each_r) * mask_te.sum()
    mse_oracle /= len(y_test)

    print(f"  Single Best Model MSE:    {mse_single_best:.4f}")
    print(f"  Global BMA (K=1) MSE:     {mse_global:.4f}")
    print(f"  Regionwise BMA (K=2) MSE: {mse_rw:.4f}")
    print(f"  Oracle Upper Bound MSE:   {mse_oracle:.4f}")
    print(f"  Improvement over Global:  {(mse_global - mse_rw)/mse_global*100:.1f}%")

    # Print weight vectors
    print(f"\n  Global weights: {dict(zip(names, np.round(w_global, 3)))}")
    for rn in ["Region1", "Region2"]:
        print(f"  {rn} weights:    {dict(zip(names, np.round(results[rn][1], 3)))}")

    return {
        "single_best": mse_single_best,
        "global_bma": mse_global,
        "regionwise_bma": mse_rw,
        "oracle": mse_oracle,
        "w_global": dict(zip(names, w_global.tolist())),
        "w_region1": dict(zip(names, results["Region1"][1].tolist())),
        "w_region2": dict(zip(names, results["Region2"][1].tolist())),
    }


# ============================================================
# Experiment 2: gamma-tau Diagnostic (C2)
# ============================================================
def exp2_gamma_tau():
    print("\n" + "=" * 60)
    print("Exp 2: gamma-tau Diagnostic Criterion (C2)")
    print("=" * 60)

    X, y_reg, y_cls, mask1, mask2 = synth_two_region(n=3000, d=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.3, random_state=SEED
    )

    # Two models: linear (good for region1) and nonlinear (good for region2)
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    model_A = Ridge(alpha=1.0, random_state=SEED)  # Linear
    model_B = RandomForestRegressor(n_estimators=100, random_state=SEED)  # Nonlinear
    model_A.fit(X_train, y_train)
    model_B.fit(X_train, y_train)

    pred_A = model_A.predict(X_train)
    pred_B = model_B.predict(X_train)
    pred_A_te = model_A.predict(X_test)
    pred_B_te = model_B.predict(X_test)

    # Compute gamma (error correlation) in sliding windows along x1
    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    gamma_vals = []
    risk_diff_vals = []

    residuals_A = pred_A - y_train
    residuals_B = pred_B - y_train

    for xv in x1_vals:
        window = (np.abs(X_train[:, 0] - xv) < 1.0)
        if window.sum() < 20:
            gamma_vals.append(np.nan)
            risk_diff_vals.append(np.nan)
            continue
        eA = residuals_A[window]
        eB = residuals_B[window]
        varA = np.var(eA)
        varB = np.var(eB)
        covAB = np.mean(eA * eB)
        if varA * varB > 0:
            gamma = covAB / np.sqrt(varA * varB)
        else:
            gamma = np.nan
        gamma_vals.append(gamma)

        # Risk difference: R_G - R_mix (global vs mixture benefit)
        mse_A = np.mean(eA ** 2)
        mse_B = np.mean(eB ** 2)
        # Mixture risk at optimal weights
        Sigma = np.array([[varA, covAB], [covAB, varB]])
        w = min_variance_weights(Sigma)
        mix_pred = w[0] * pred_A[window] + w[1] * pred_B[window]
        mse_mix = np.mean((mix_pred - y_train[window]) ** 2)
        risk_diff_vals.append(mse_A - mse_mix)  # Mixing benefit

    gamma_vals = np.array(gamma_vals)
    risk_diff_vals = np.array(risk_diff_vals)

    print("  Sliding window analysis along x1:")
    print(f"  x1 range: [{x1_vals[0]:.2f}, {x1_vals[-1]:.2f}]")
    print(f"  Mean gamma: {np.nanmean(gamma_vals):.3f}")
    print(f"  Mean mixing benefit (R_G - R_mix): {np.nanmean(risk_diff_vals):.4f}")

    # Correlation between gamma and mixing benefit
    valid = ~np.isnan(gamma_vals) & ~np.isnan(risk_diff_vals)
    if valid.sum() > 5:
        corr = np.corrcoef(gamma_vals[valid], risk_diff_vals[valid])[0, 1]
        print(f"  Correlation(gamma, mixing_benefit): {corr:.3f}")
        print(f"  (Expected: negative — lower gamma → higher mixing benefit)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(x1_vals, gamma_vals, 'b-o', markersize=3)
    ax1.axvline(x=0, color='r', linestyle='--', label='True boundary (x1=0)')
    ax1.set_ylabel('gamma (error corr)')
    ax1.set_title('gamma-tau Diagnostic: gamma vs Mixing Benefit along x1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x1_vals, risk_diff_vals, 'g-o', markersize=3)
    ax2.axvline(x=0, color='r', linestyle='--', label='True boundary (x1=0)')
    ax2.set_ylabel('Mixing Benefit\n(R_G - R_mix)')
    ax2.set_xlabel('x1 (input coordinate)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_gamma_tau.png"), dpi=150)
    plt.close()
    print(f"  Saved: fig2_gamma_tau.png")

    return {"gamma_corr": corr if valid.sum() > 5 else None}


# ============================================================
# Experiment 3: Minority Collapse + Gate (C3)
# ============================================================
def exp3_minority_collapse():
    print("\n" + "=" * 60)
    print("Exp 3: Minority-Region Weight Collapse + Discrete-Event Gate (C3)")
    print("=" * 60)

    results = synth_minority_collapse(n_majority=1800,
                                      n_minority_list=[500, 200, 100, 50, 20, 10])

    n_min_list = sorted(results.keys())
    w_local_no_gate = []
    w_local_with_gate = []
    minority_f1_no_gate = []
    minority_f1_gate = []

    from sklearn.linear_model import Ridge

    for n_min in n_min_list:
        X, y, mask_min, n_maj = results[n_min]
        X_train, X_test, y_train, y_test, mm_tr, mm_te = train_test_split(
            X, y, mask_min, test_size=0.3, random_state=SEED
        )

        # Global model (trained on all data)
        global_model = Ridge(alpha=1.0, random_state=SEED)
        global_model.fit(X_train, y_train)

        # Local model (trained on minority region only)
        local_model = Ridge(alpha=1.0, random_state=SEED)
        if mm_tr.sum() >= 5:
            local_model.fit(X_train[mm_tr], y_train[mm_tr])
        else:
            local_model.fit(X_train[:5], y_train[:5])  # fallback

        # Predictions
        pred_global = global_model.predict(X_train)
        pred_local = local_model.predict(X_train)
        pred_global_te = global_model.predict(X_test)
        pred_local_te = local_model.predict(X_test)

        # Estimate Sigma on minority region
        eG = pred_global[mm_tr] - y_train[mm_tr]
        eL = pred_local[mm_tr] - y_train[mm_tr]
        a = np.var(eG)
        b = np.var(eL)
        c = np.mean(eG * eL)
        Sigma = np.array([[a, c], [c, b]])

        # Without gate
        w_no_gate = min_variance_weights(Sigma)
        # With gate (eps_min = 0.1)
        w_gate = min_variance_weights(Sigma, eps_min=0.1)

        w_local_no_gate.append(w_no_gate[1])
        w_local_with_gate.append(w_gate[1])

        # Minority region performance on test
        if mm_te.sum() > 0:
            # Predictions on minority test
            mix_no_gate = w_no_gate[0] * pred_global_te[mm_te] + w_no_gate[1] * pred_local_te[mm_te]
            mix_gate = w_gate[0] * pred_global_te[mm_te] + w_gate[1] * pred_local_te[mm_te]
            # Use R^2 as performance metric
            from sklearn.metrics import r2_score
            r2_no_gate = r2_score(y_test[mm_te], mix_no_gate)
            r2_gate = r2_score(y_test[mm_te], mix_gate)
        else:
            r2_no_gate = 0
            r2_gate = 0

        minority_f1_no_gate.append(r2_no_gate)
        minority_f1_gate.append(r2_gate)

        print(f"  n_min={n_min:4d}: w_L(no gate)={w_no_gate[1]:.4f}, "
              f"w_L(gate)={w_gate[1]:.4f}, "
              f"R2 no-gate={r2_no_gate:.3f}, R2 gate={r2_gate:.3f}")

    # Plot collapse curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(n_min_list, w_local_no_gate, 'ro-', label='No gate', markersize=5)
    ax1.plot(n_min_list, w_local_with_gate, 'bs--', label='With gate (eps=0.1)', markersize=5)
    ax1.set_xscale('log')
    ax1.set_xlabel('n_min (minority region sample size)')
    ax1.set_ylabel('w_L (local model weight)')
    ax1.set_title('Weight Collapse vs n_min')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.plot(n_min_list, minority_f1_no_gate, 'ro-', label='No gate', markersize=5)
    ax2.plot(n_min_list, minority_f1_gate, 'bs--', label='With gate (eps=0.1)', markersize=5)
    ax2.set_xscale('log')
    ax2.set_xlabel('n_min (minority region sample size)')
    ax2.set_ylabel('R^2 on minority test')
    ax2.set_title('Minority Region Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_collapse_gate.png"), dpi=150)
    plt.close()
    print(f"  Saved: fig3_collapse_gate.png")

    return {
        "n_min_list": n_min_list,
        "w_no_gate": w_local_no_gate,
        "w_gate": w_local_with_gate,
        "r2_no_gate": minority_f1_no_gate,
        "r2_gate": minority_f1_gate,
    }


# ============================================================
# Experiment 4: Subset Selection Efficiency (C4)
# ============================================================
def exp4_subset_selection():
    print("\n" + "=" * 60)
    print("Exp 4: Sparse Setwise BMA — Greedy Subset Selection (C4)")
    print("=" * 60)

    X, y = synth_m_model(n=1500, d=8, M=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )

    # Create M=20 models: varied regularization strengths
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    M = 10  # Use M=10 for exhaustive search feasibility
    models = []
    names = []
    for alpha in np.logspace(-2, 2, 5):
        m = Ridge(alpha=alpha, random_state=SEED)
        models.append(m)
        names.append(f"Ridge_{alpha:.1f}")
    for n_est in [10, 50]:
        m = RandomForestRegressor(n_estimators=n_est, random_state=SEED)
        models.append(m)
        names.append(f"RF_{n_est}")
    for lr in [0.01, 0.1, 0.5]:
        m = GradientBoostingRegressor(n_estimators=20, learning_rate=lr, random_state=SEED)
        models.append(m)
        names.append(f"GB_{lr}")

    M = len(models)
    for m in models:
        m.fit(X_train, y_train)

    preds_train = np.column_stack([m.predict(X_train) for m in models])
    preds_test = np.column_stack([m.predict(X_test) for m in models])
    residuals = preds_train - y_train[:, None]
    Sigma = compute_error_covariance(residuals)

    # Function: minimum variance for a given subset
    def min_var(subset):
        if len(subset) == 0:
            return np.inf
        Sigma_sub = Sigma[np.ix_(subset, subset)]
        w = min_variance_weights(Sigma_sub)
        return w @ Sigma_sub @ w

    # Exhaustive search (M=10 → 2^10 = 1024 subsets)
    from itertools import combinations
    best_val = np.inf
    best_set = None
    all_subset_results = []

    for r in range(1, M + 1):
        for combo in combinations(range(M), r):
            val = min_var(list(combo))
            all_subset_results.append((list(combo), val))
            if val < best_val:
                best_val = val
                best_set = list(combo)

    print(f"  M = {M} models, exhaustive search: {len(all_subset_results)} subsets")
    print(f"  Optimal subset size: {len(best_set)}")
    print(f"  Optimal subset: {[names[i] for i in best_set]}")
    print(f"  Optimal minimum variance: {best_val:.6f}")

    # Greedy forward selection
    greedy_set = []
    greedy_vals = []
    for step in range(M):
        best_improvement = 0
        best_m = None
        current_val = min_var(greedy_set) if greedy_set else np.inf
        for m in range(M):
            if m in greedy_set:
                continue
            trial = greedy_set + [m]
            val = min_var(trial)
            improvement = current_val - val if greedy_set else current_val - val
            if improvement > best_improvement or best_m is None:
                best_improvement = improvement
                best_m = m
        if best_m is not None:
            greedy_set.append(best_m)
            greedy_vals.append(min_var(greedy_set))

    print(f"\n  Greedy selection:")
    print(f"  Greedy subset size: {len(greedy_set)}")
    print(f"  Greedy subset: {[names[i] for i in greedy_set]}")
    print(f"  Greedy minimum variance: {min_var(greedy_set):.6f}")

    # Compare: greedy / optimal ratio
    ratio = min_var(greedy_set) / best_val
    print(f"  Greedy/Optimal ratio: {ratio:.4f}")
    print(f"  (Expected near 1.0 for greedy forward selection)")

    # Test set performance
    w_opt = min_variance_weights(Sigma[np.ix_(best_set, best_set)])
    pred_opt = preds_test[:, best_set] @ w_opt
    mse_opt = np.mean((pred_opt - y_test) ** 2)

    w_greedy = min_variance_weights(Sigma[np.ix_(greedy_set, greedy_set)])
    pred_greedy = preds_test[:, greedy_set] @ w_greedy
    mse_greedy = np.mean((pred_greedy - y_test) ** 2)

    # All models (no selection)
    w_all = min_variance_weights(Sigma)
    pred_all = preds_test @ w_all
    mse_all = np.mean((pred_all - y_test) ** 2)

    print(f"\n  Test MSE — All models:    {mse_all:.4f}")
    print(f"  Test MSE — Optimal subset:{mse_opt:.4f}")
    print(f"  Test MSE — Greedy subset: {mse_greedy:.4f}")

    return {
        "M": M,
        "optimal_size": len(best_set),
        "greedy_size": len(greedy_set),
        "greedy_optimal_ratio": ratio,
        "mse_all": mse_all,
        "mse_optimal": mse_opt,
        "mse_greedy": mse_greedy,
    }


# ============================================================
# Experiment 7: K vs Regret U-Curve
# ============================================================
def exp7_k_regret_curve():
    print("\n" + "=" * 60)
    print("Exp 7: K vs Total Regret (U-Curve)")
    print("=" * 60)

    X, y_reg, y_cls, mask1, mask2 = synth_two_region(n=3000, d=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.3, random_state=SEED
    )

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    # Two models: linear and nonlinear
    model_A = Ridge(alpha=1.0, random_state=SEED)
    model_B = RandomForestRegressor(n_estimators=50, random_state=SEED)
    model_A.fit(X_train, y_train)
    model_B.fit(X_train, y_train)

    pred_A = model_A.predict(X_train)
    pred_B = model_B.predict(X_train)
    pred_A_te = model_A.predict(X_test)
    pred_B_te = model_B.predict(X_test)

    # Vary K: split x1 axis into K regions
    K_values = [1, 2, 3, 5, 8, 12, 20, 50, 100]
    total_mse_list = []
    boundary_mse_list = []
    within_region_mse_list = []

    x1_train = X_train[:, 0]
    x1_test = X_test[:, 0]

    for K in K_values:
        if K == 1:
            # Global BMA
            Sigma = compute_error_covariance(
                np.column_stack([pred_A - y_train, pred_B - y_train])
            )
            w = min_variance_weights(Sigma)
            pred = w[0] * pred_A_te + w[1] * pred_B_te
            total_mse = np.mean((pred - y_test) ** 2)
            total_mse_list.append(total_mse)
            within_region_mse_list.append(total_mse)
            boundary_mse_list.append(0)
            continue

        # Create K bins along x1
        edges = np.quantile(x1_train, np.linspace(0, 1, K + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf

        total_se = 0
        boundary_se = 0
        within_se = 0
        n_boundary = 0
        n_total = 0

        for k in range(K):
            mask_tr = (x1_train >= edges[k]) & (x1_train < edges[k + 1])
            mask_te = (x1_test >= edges[k]) & (x1_test < edges[k + 1])
            if mask_tr.sum() < 10 or mask_te.sum() < 3:
                continue

            eA = pred_A[mask_tr] - y_train[mask_tr]
            eB = pred_B[mask_tr] - y_train[mask_tr]
            Sigma = np.array([[np.var(eA), np.mean(eA*eB)],
                              [np.mean(eA*eB), np.var(eB)]])
            w = min_variance_weights(Sigma)

            pred_r = w[0] * pred_A_te[mask_te] + w[1] * pred_B_te[mask_te]
            se = np.sum((pred_r - y_test[mask_te]) ** 2)
            total_se += se
            within_se += se
            n_total += mask_te.sum()

            # Boundary samples (within 5% of bin edge)
            if k > 0:
                mask_b = (x1_test >= edges[k] - 0.1) & (x1_test < edges[k] + 0.1)
                if mask_b.sum() > 0:
                    pred_b = w[0] * pred_A_te[mask_b] + w[1] * pred_B_te[mask_b]
                    boundary_se += np.sum((pred_b - y_test[mask_b]) ** 2)
                    n_boundary += mask_b.sum()

        total_mse = total_se / max(n_total, 1)
        within_mse = within_se / max(n_total, 1)
        boundary_mse = boundary_se / max(n_boundary, 1) if n_boundary > 0 else 0

        total_mse_list.append(total_mse)
        within_region_mse_list.append(within_mse)
        boundary_mse_list.append(boundary_mse)

        print(f"  K={K:3d}: total_MSE={total_mse:.4f}, "
              f"within={within_mse:.4f}, boundary={boundary_mse:.4f}")

    # Plot U-curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_values, total_mse_list, 'ro-', label='Total MSE', markersize=5)
    ax.plot(K_values, within_region_mse_list, 'b--', label='Within-region MSE', alpha=0.7)
    ax.plot(K_values, boundary_mse_list, 'g:', label='Boundary MSE', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('K (number of regions)')
    ax.set_ylabel('MSE')
    ax.set_title('K vs Total Regret — Expected U-Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig6_k_regret.png"), dpi=150)
    plt.close()
    print(f"  Saved: fig6_k_regret.png")

    return {
        "K_values": K_values,
        "total_mse": total_mse_list,
        "within_mse": within_region_mse_list,
        "boundary_mse": boundary_mse_list,
    }


# ============================================================
# Run All Experiments
# ============================================================
if __name__ == "__main__":
    print("Regionwise BMA — Core Experiments")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print()

    results = {}
    results["exp1"] = exp1_closed_form_weights()
    results["exp2"] = exp2_gamma_tau()
    results["exp3"] = exp3_minority_collapse()
    results["exp4"] = exp4_subset_selection()
    results["exp7"] = exp7_k_regret_curve()

    # Save summary
    summary_path = os.path.join(OUT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")
    print("\nAll experiments complete!")
