"""
Paper method column rerun: gamma-tau split + discrete-event gate (eps_min=0.1)
==============================================================================
Faithful implementation of the paper's Regionwise BMA method (Sec 3.3-3.4):
  - Region partition: gamma-tau driven recursive partitioning (max_depth=2,
    tau=0.005, min_leaf=20, feature-quantile search, log-loss benefit), the
    same implementation as exp_supplementary.py.
  - Per-region weights: min-variance inverse-covariance BMA (min_var_w),
    estimated on the inner-validation residuals (same protocol as
    sota_baselines.py).
  - Gate: discrete-event gate lifting the smallest regional weight to
    eps_min=0.1 and renormalizing (Theorem 3; exp_supplementary logic).
  - Fallback to global BMA when a region has <15 validation or <3 test
    samples (same rule as sota_baselines.py).

Protocol: 70/30 stratified train/test, 10 splits, seeds 42+10*s (s=0..9),
base pool LR/RF(40,d6)/GB(25,d3)/SVM(prob)/kNN(k=7), metric AUC,
capped at 5000 samples. Identical to sota_baselines.py.

Global / Stacking / Deep Ensemble are reused from figures/sota_baselines.json
(same protocol). This script only reruns the Regionwise (paper method) column.

Usage: python run_paper_method.py [--smoke]
"""
import numpy as np
import pandas as pd
from scipy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os, json, sys, glob, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sota_baselines import (SEED, N_SPLITS, MAX_SAMPLES, REGIME, DATA_DIR, DATA2_DIR,
                            min_var_w, load_csv, parse_keel_dat, load_zip, get_models,
                            collect_datasets)

OUT_DIR = r"C:\Users\robot\Desktop\M PAPER\figures"
os.makedirs(OUT_DIR, exist_ok=True)

EPS_MIN = 0.10          # gate floor (paper Sec 4.8 / sensitivity)
TAU = 0.005             # gamma-tau stopping threshold (exp_supplementary default)
MAX_DEPTH = 2           # gamma-tau tree depth
MIN_LEAF = 20           # minimum samples per leaf (gamma-tau + region rule)


def gamma_tau_split(X, P, y, max_depth=MAX_DEPTH, tau=TAU, min_leaf=MIN_LEAF):
    """Gamma-tau driven recursive partitioning (paper Sec 3.3).

    Recursively splits on the feature-quantile point that maximizes
    log-loss benefit (global risk - weighted regional risk); stops when the
    best benefit < tau or depth/size limits are reached. Identical to
    exp_supplementary.py::gamma_tau_split.
    """
    def _split(X, P, y, depth):
        n = len(y)
        if depth >= max_depth or n < 2 * min_leaf:
            return {"type": "leaf"}

        best_benefit = 0.0
        best_feat = 0
        best_thresh = np.median(X[:, 0])

        res_g = P - y[:, None]
        w_g = min_var_w(np.cov(res_g, rowvar=False, bias=True))
        pred_g = np.clip(P @ w_g, 1e-7, 1 - 1e-7)
        R_g = -np.mean(y * np.log(pred_g) + (1 - y) * np.log(1 - pred_g))

        for feat in range(X.shape[1]):
            order = np.argsort(X[:, feat])
            x_sorted = X[order, feat]
            for q in range(2, 10):
                idx = int(n * q / 10)
                if idx < min_leaf or n - idx < min_leaf:
                    continue
                thresh = (x_sorted[idx - 1] + x_sorted[idx]) / 2
                left = X[:, feat] <= thresh
                right = ~left
                if left.sum() < min_leaf or right.sum() < min_leaf:
                    continue
                try:
                    w_l = min_var_w(np.cov(P[left] - y[left, None], rowvar=False, bias=True))
                    pred_l = np.clip(P[left] @ w_l, 1e-7, 1 - 1e-7)
                    R_l = -np.mean(y[left] * np.log(pred_l) + (1 - y[left]) * np.log(1 - pred_l))

                    w_r = min_var_w(np.cov(P[right] - y[right, None], rowvar=False, bias=True))
                    pred_r = np.clip(P[right] @ w_r, 1e-7, 1 - 1e-7)
                    R_r = -np.mean(y[right] * np.log(pred_r) + (1 - y[right]) * np.log(1 - pred_r))

                    R_weighted = left.sum() / n * R_l + right.sum() / n * R_r
                    benefit = R_g - R_weighted
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_feat = feat
                        best_thresh = thresh
                except Exception:
                    continue

        if best_benefit < tau:
            return {"type": "leaf"}
        return {
            "type": "split", "feat": best_feat, "thresh": best_thresh,
            "left": _split(X[X[:, best_feat] <= best_thresh],
                          P[X[:, best_feat] <= best_thresh],
                          y[X[:, best_feat] <= best_thresh], depth + 1),
            "right": _split(X[X[:, best_feat] > best_thresh],
                           P[X[:, best_feat] > best_thresh],
                           y[X[:, best_feat] > best_thresh], depth + 1),
        }

    return _split(X, P, y, 0)


def get_region_labels(tree, X):
    """Region assignment for every row of X (applies the trained tree)."""
    n = len(X)
    labels = np.zeros(n, dtype=int)

    def _assign(node, mask, region_id):
        if node["type"] == "leaf":
            labels[mask] = region_id
            return region_id + 1
        feat, thresh = node["feat"], node["thresh"]
        left_mask = mask & (X[:, feat] <= thresh)
        right_mask = mask & (X[:, feat] > thresh)
        nid = _assign(node["left"], left_mask, region_id)
        return _assign(node["right"], right_mask, nid)

    _assign(tree, np.ones(n, dtype=bool), 0)
    return labels


def gate_w(w, eps_min=EPS_MIN):
    """Discrete-event gate: lift the smallest regional weight to eps_min.

    Corresponds to the constrained optimization in Theorem 3 (w_L >= eps_min
    active at the boundary): the collapsed expert is restored to the floor
    and weights are renormalized.
    """
    w2 = w.copy()
    i_min = int(np.argmin(w2))
    if w2[i_min] < eps_min:
        w2[i_min] = eps_min
        s = w2.sum()
        if s > 0:
            w2 = w2 / s
    return w2


def run_single_split(X, y, seed):
    """Return AUC dict for global and paper-method regionwise on one split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    models = get_models()
    P_tr_list, P_te_list, names = [], [], []
    for name, mdl in models:
        try:
            mdl.fit(X_tr, y_tr)
            P_tr_list.append(mdl.predict_proba(X_tr)[:, 1])
            P_te_list.append(mdl.predict_proba(X_te)[:, 1])
            names.append(name)
        except Exception:
            pass
    if len(P_tr_list) < 3:
        return None
    P_tr = np.column_stack(P_tr_list)
    P_te = np.column_stack(P_te_list)
    M = len(names)

    # Inner validation split (same protocol as BMA covariance estimation)
    X_t2, X_val, y_t2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.3, random_state=seed, stratify=y_tr)
    P_val_list = []
    for i in range(M):
        try:
            mdl = get_models()[i][1]
            mdl.fit(X_t2, y_t2)
            P_val_list.append(mdl.predict_proba(X_val)[:, 1])
        except Exception:
            P_val_list.append(P_tr_list[i][:len(X_val)])
    P_val = np.column_stack(P_val_list)

    # 1. Global BMA
    res = P_val - y_val[:, None]
    Sigma = np.cov(res, rowvar=False, bias=True)
    w_g = min_var_w(Sigma)
    auc_global = roc_auc_score(y_te, P_te @ w_g)

    # 2. Paper method: median split on first feature (Sec 4.1 protocol) + gate.
    #    This is the region partition used by the main experiments (Table 5);
    #    the gamma-tau split below is the Sec 4.6 comparison variant.
    median_x1 = np.median(X_tr[:, 0])
    rv = X_val[:, 0] > median_x1
    rt = X_te[:, 0] > median_x1
    pred_med = np.zeros(len(y_te))
    pred_med_gate = np.zeros(len(y_te))
    for rmask, tmask in [(rv, rt), (~rv, ~rt)]:
        if rmask.sum() < 15 or tmask.sum() < 3:
            pred_med[tmask] = P_te[tmask] @ w_g
            pred_med_gate[tmask] = P_te[tmask] @ w_g
            continue
        res_r = P_val[rmask] - y_val[rmask, None]
        Sigma_r = np.cov(res_r, rowvar=False, bias=True)
        w_r = min_var_w(Sigma_r)
        pred_med[tmask] = P_te[tmask] @ w_r
        pred_med_gate[tmask] = P_te[tmask] @ gate_w(w_r)

    # 3. Gamma-tau split variant (Sec 4.6): tree built on train, in-sample probs
    tree = gamma_tau_split(X_tr, P_tr, y_tr)
    rv_gt = get_region_labels(tree, X_val)
    rt_gt = get_region_labels(tree, X_te)

    pred_ng = np.zeros(len(y_te))
    pred_g = np.zeros(len(y_te))
    for k in range(int(rv_gt.max()) + 1):
        rmask = rv_gt == k
        tmask = rt_gt == k
        if rmask.sum() < 15 or tmask.sum() < 3:
            pred_ng[tmask] = P_te[tmask] @ w_g
            pred_g[tmask] = P_te[tmask] @ w_g
            continue
        res_r = P_val[rmask] - y_val[rmask, None]
        Sigma_r = np.cov(res_r, rowvar=False, bias=True)
        w_r = min_var_w(Sigma_r)
        pred_ng[tmask] = P_te[tmask] @ w_r
        pred_g[tmask] = P_te[tmask] @ gate_w(w_r)

    return {
        "global": auc_global,
        "rw_gt": roc_auc_score(y_te, pred_ng),
        "rw_gt_gate": roc_auc_score(y_te, pred_g),
        "rw_med": roc_auc_score(y_te, pred_med),
        "rw_med_gate": roc_auc_score(y_te, pred_med_gate),
    }


def collect_all():
    """Collect paper datasets: data/*.csv + data2 zips at 2 and 3 levels deep.

    sota_baselines.collect_datasets only scans 2 levels, missing the
    poker/winequality zips under imb_IRHigherThan9p3/ sub-subdirectories.
    """
    datasets = {}
    seen = set()

    def norm(n):
        return n.lower().replace("_", "").replace("-", "")

    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        dname = os.path.basename(fpath).replace(".csv", "")
        key = norm(dname)
        if key in seen:
            continue
        try:
            data = load_csv(fpath)
            if data is None:
                continue
            datasets[dname] = data
            seen.add(key)
        except Exception:
            continue

    # 2-level: data2/<group>/<file>.zip ; 3-level: data2/<group>/<sub>/<file>.zip
    for dirs in [("*",), ("*", "*")]:
        for zpath in sorted(glob.glob(os.path.join(DATA2_DIR, *dirs, "*.zip"))):
            if "fold" in os.path.basename(zpath).lower():
                continue
            # KEEL zips: <dataset>.zip (2-level) or <dir>/<dataset>.zip (3-level);
            # name the dataset after the zip basename, not the parent dir
            dname = os.path.basename(zpath)[:-4]
            key = norm(dname)
            if key in seen:
                continue
            try:
                data = load_zip(zpath)
                if data is None:
                    continue
                datasets[dname] = data
                seen.add(key)
            except Exception:
                continue
    return datasets


def main():
    smoke = "--smoke" in sys.argv
    n_splits = 1 if smoke else N_SPLITS
    print(f"mode: {'SMOKE (1 split)' if smoke else 'FULL (10 splits)'}")

    datasets = collect_all()
    paper_names = [k for k in datasets if k in REGIME]
    paper_names = sorted(paper_names, key=lambda k: REGIME[k])
    print(f"paper datasets found: {len(paper_names)}")

    results = {}
    for dname in paper_names:
        X, y = datasets[dname]
        n, d = X.shape
        n_pos = int(y.sum())
        ir = n_pos / (n - n_pos) if (n - n_pos) > 0 else 999.0
        regime = REGIME[dname]
        split_results = []
        for s in range(n_splits):
            res = run_single_split(X, y, seed=SEED + s * 10)
            if res is not None:
                split_results.append(res)
        if len(split_results) < (1 if smoke else 3):
            print(f"  SKIP {dname} (only {len(split_results)} valid splits)")
            continue
        agg = {"n": n, "d": d, "ir": round(float(ir), 4), "regime": regime,
               "n_splits": len(split_results)}
        for m in ["global", "rw_gt", "rw_gt_gate", "rw_med", "rw_med_gate"]:
            vals = np.array([r[m] for r in split_results])
            agg[m + "_mean"] = round(float(vals.mean()), 4)
            agg[m + "_std"] = round(float(vals.std(ddof=1)), 4)
        results[dname] = agg
        print(f"  {dname:24s} G={agg['global_mean']:.4f} "
              f"MED={agg['rw_med_mean']:.4f} MED_GATE={agg['rw_med_gate_mean']:.4f} "
              f"GT={agg['rw_gt_mean']:.4f} ({regime})")

    with open(os.path.join(OUT_DIR, "paper_method_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mode": "smoke" if smoke else "full",
                   "eps_min": EPS_MIN, "tau": TAU,
                   "n_datasets": len(results), "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_DIR}/paper_method_results.json  ({len(results)} datasets)")


if __name__ == "__main__":
    main()
