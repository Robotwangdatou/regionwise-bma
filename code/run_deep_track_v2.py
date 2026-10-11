"""
Deep Track v2: CIFAR-10 ResNet-18 features
5 seeds x 2 pairs x (Global / Ungated RW / Gated RW)
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.linalg import inv
from scipy import stats
import os, json, time

OUT_DIR = r"C:\Users\robot\Desktop\M PAPER\figures"
os.makedirs(OUT_DIR, exist_ok=True)
FEAT_CACHE = os.path.join(OUT_DIR, "cifar10_resnet18_features.npz")

# ============================================================
# Step 1: Extract features once, cache to disk
# ============================================================
if os.path.exists(FEAT_CACHE):
    print("Loading cached features...")
    d = np.load(FEAT_CACHE)
    X, y = d['X'], d['y']
    print(f"Loaded: X={X.shape}, y={y.shape}")
else:
    print("Extracting ResNet-18 features (first time, ~2 min)...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    n_subset = 10000
    rng = np.random.RandomState(42)
    indices = rng.choice(len(train_dataset), n_subset, replace=False)

    features, labels = [], []
    batch_size = 64
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_subset, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_images = torch.stack([train_dataset[j][0] for j in batch_idx])
            batch_labels = [train_dataset[j][1] for j in batch_idx]
            feat = model(batch_images).squeeze(-1).squeeze(-1).numpy()
            features.append(feat)
            labels.extend(batch_labels)
            if (i // batch_size) % 20 == 0:
                print(f"  {i+batch_size}/{n_subset} ({time.time()-t0:.1f}s)")

    X = np.vstack(features)
    y = np.array(labels)
    np.savez(FEAT_CACHE, X=X, y=y)
    print(f"Features saved: {X.shape}")

# ============================================================
# Step 2: Run one configuration
# ============================================================
def run_one(majority_class, minority_class, seed, n_maj=2000, target_ir=0.1, d_pca=50, eps_min=0.1):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    maj_idx = np.where(y == majority_class)[0]
    min_idx = np.where(y == minority_class)[0]

    # Subsample to long-tailed
    if len(maj_idx) > n_maj:
        maj_idx = rng.choice(maj_idx, n_maj, replace=False)
    target_min = int(len(maj_idx) * target_ir)
    if len(min_idx) > target_min:
        min_idx = rng.choice(min_idx, target_min, replace=False)

    X_lt = np.vstack([X[maj_idx], X[min_idx]])
    y_lt = np.hstack([np.zeros(len(maj_idx)), np.ones(len(min_idx))])

    # PCA to d_pca
    pca = PCA(n_components=d_pca, random_state=seed)
    X_pca = pca.fit_transform(X_lt)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_lt, test_size=0.3, random_state=seed, stratify=y_lt
    )

    # Base models
    base_models = [
        LogisticRegression(max_iter=1000, random_state=seed),
        RandomForestClassifier(n_estimators=50, random_state=seed),
        GradientBoostingClassifier(n_estimators=50, random_state=seed),
        SVC(kernel='rbf', probability=True, random_state=seed),
        KNeighborsClassifier(n_neighbors=7),
    ]
    M = len(base_models)

    preds_train = np.zeros((len(X_train), M))
    preds_test = np.zeros((len(X_test), M))
    for i, m in enumerate(base_models):
        m.fit(X_train, y_train)
        preds_train[:, i] = m.predict_proba(X_train)[:, 1]
        preds_test[:, i] = m.predict_proba(X_test)[:, 1]

    errors = preds_train - y_train[:, None]

    # Global BMA
    Sigma_g = np.cov(errors.T)
    inv_Sg = inv(Sigma_g + 1e-6 * np.eye(M))
    ones = np.ones(M)
    w_g = inv_Sg @ ones / (ones @ inv_Sg @ ones)
    auc_global = roc_auc_score(y_test, preds_test @ w_g)

    # PCA-median split
    pca_split = PCA(n_components=1, random_state=seed)
    pc1_tr = pca_split.fit_transform(X_train).flatten()
    pc1_te = pca_split.transform(X_test).flatten()
    med = np.median(pc1_tr)
    reg_tr = (pc1_tr > med).astype(int)
    reg_te = (pc1_te > med).astype(int)

    # Regionwise BMA (no gate)
    pred_nogate = np.zeros(len(y_test))
    w_nogate_list = []
    for k in range(2):
        mask_tr = reg_tr == k
        mask_te = reg_te == k
        n_k = mask_tr.sum()
        if n_k < 20:
            w_k = w_g.copy()
        else:
            Sigma_k = np.cov(errors[mask_tr].T)
            inv_Sk = inv(Sigma_k + 1e-6 * np.eye(M))
            w_k = inv_Sk @ ones / (ones @ inv_Sk @ ones)
        w_nogate_list.append(w_k)
        pred_nogate[mask_te] = preds_test[mask_te] @ w_k

    auc_nogate = roc_auc_score(y_test, pred_nogate)

    # Regionwise BMA + Gate
    pred_gate = np.zeros(len(y_test))
    min_weight_nogate = 1.0
    ceiwc_triggers = 0
    for k in range(2):
        mask_te = reg_te == k
        w_k = w_nogate_list[k].copy()
        mw = w_k.min()
        if mw < min_weight_nogate:
            min_weight_nogate = mw
        if mw < eps_min:
            ceiwc_triggers += 1
        w_gated = np.clip(w_k, eps_min, None)
        w_gated = w_gated / w_gated.sum()
        pred_gate[mask_te] = preds_test[mask_te] @ w_gated

    auc_gate = roc_auc_score(y_test, pred_gate)

    return {
        'auc_global': auc_global,
        'auc_nogate': auc_nogate,
        'auc_gate': auc_gate,
        'delta_nogate': auc_nogate - auc_global,
        'delta_gate': auc_gate - auc_global,
        'gate_effect': auc_gate - auc_nogate,
        'min_weight_nogate': min_weight_nogate,
        'ceiwc_triggers': ceiwc_triggers,
    }

# ============================================================
# Step 3: Run 5 seeds x 2 pairs
# ============================================================
pairs = {
    'cat_vs_dog': (3, 5),
    'ship_vs_truck': (8, 9),
}
seeds = [42, 123, 2024, 7, 99]

all_results = {}
for pair_name, (maj_cls, min_cls) in pairs.items():
    print(f"\n{'='*60}")
    print(f"Pair: {pair_name} (majority={maj_cls}, minority={min_cls})")
    print(f"{'='*60}")
    pair_results = []
    for seed in seeds:
        t0 = time.time()
        r = run_one(maj_cls, min_cls, seed)
        pair_results.append(r)
        print(f"  seed={seed}: Global={r['auc_global']:.4f}, "
              f"RW-noGate={r['auc_nogate']:.4f} (Δ={r['delta_nogate']:+.4f}), "
              f"RW+Gate={r['auc_gate']:.4f} (Δ={r['delta_gate']:+.4f}), "
              f"gate_effect={r['gate_effect']:+.4f}, "
              f"min_w={r['min_weight_nogate']:.4f}, "
              f"triggers={r['ceiwc_triggers']}/2 ({time.time()-t0:.1f}s)")

    # Aggregate
    globals_auc = [r['auc_global'] for r in pair_results]
    nogate_auc = [r['auc_nogate'] for r in pair_results]
    gate_auc = [r['auc_gate'] for r in pair_results]
    delta_nogate = [r['delta_nogate'] for r in pair_results]
    delta_gate = [r['delta_gate'] for r in pair_results]
    gate_effect = [r['gate_effect'] for r in pair_results]
    min_w = [r['min_weight_nogate'] for r in pair_results]
    triggers = [r['ceiwc_triggers'] for r in pair_results]

    # Paired t-test: gate vs nogate
    t_stat, p_val = stats.ttest_rel(gate_auc, nogate_auc)

    summary = {
        'mean_global': float(np.mean(globals_auc)),
        'std_global': float(np.std(globals_auc)),
        'mean_nogate': float(np.mean(nogate_auc)),
        'std_nogate': float(np.std(nogate_auc)),
        'mean_gate': float(np.mean(gate_auc)),
        'std_gate': float(np.std(gate_auc)),
        'mean_delta_nogate': float(np.mean(delta_nogate)),
        'std_delta_nogate': float(np.std(delta_nogate)),
        'mean_delta_gate': float(np.mean(delta_gate)),
        'std_delta_gate': float(np.std(delta_gate)),
        'mean_gate_effect': float(np.mean(gate_effect)),
        'std_gate_effect': float(np.std(gate_effect)),
        'ttest_gate_vs_nogate_t': float(t_stat),
        'ttest_gate_vs_nogate_p': float(p_val),
        'mean_min_weight_nogate': float(np.mean(min_w)),
        'total_ceiwc_triggers': int(sum(triggers)),
        'per_seed': pair_results,
    }
    all_results[pair_name] = summary

    print(f"\n  --- SUMMARY ({pair_name}) ---")
    print(f"  Global BMA:    {np.mean(globals_auc):.4f} ± {np.std(globals_auc):.4f}")
    print(f"  RW no-gate:    {np.mean(nogate_auc):.4f} ± {np.std(nogate_auc):.4f}  (Δ={np.mean(delta_nogate):+.4f})")
    print(f"  RW + Gate:     {np.mean(gate_auc):.4f} ± {np.std(gate_auc):.4f}  (Δ={np.mean(delta_gate):+.4f})")
    print(f"  Gate effect:   {np.mean(gate_effect):+.4f} ± {np.std(gate_effect):.4f}  (paired t={t_stat:.3f}, p={p_val:.4f})")
    print(f"  Mean min weight (no gate): {np.mean(min_w):.4f}")
    print(f"  CEIWC triggers: {sum(triggers)}/10 regions across 5 seeds")

# Save
with open(os.path.join(OUT_DIR, 'cifar10_deep_track_v2_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n\nResults saved to cifar10_deep_track_v2_results.json")
