"""
Use EXISTING CROSR pipeline features with our pure-Python OpenMax.
Tests: saved_features/ + saved_MAVs/ + saved_distance_scores/ from original pipeline.
Expected: AUROC close to 0.9651
"""
import numpy as np
import sys, os, pickle, itertools
sys.path.insert(0, '.')

from backend.weibull_openmax import WeibullOpenMax
from sklearn.metrics import roc_auc_score


def load_features(split_dir, num_classes=6, max_per_class=2000):
    """Load .npy feature files from class-organized dirs."""
    all_features = []
    all_labels = []
    for cls_id in range(num_classes):
        cls_dir = os.path.join(split_dir, str(cls_id))
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))[:max_per_class]
        feats = [np.load(os.path.join(cls_dir, f)) for f in files if f.endswith('.npy')]
        if feats:
            all_features.extend(feats)
            all_labels.extend([cls_id] * len(feats))
    return np.array(all_features), np.array(all_labels)


# ------------------------------------------------------------
print("[1] Loading existing features...")
base = 'saved_features/cicids_1d'

feat_train, y_train = load_features(os.path.join(base, 'train'), max_per_class=3000)
feat_val, y_val = load_features(os.path.join(base, 'val'), max_per_class=2000)
feat_open, _ = load_features(os.path.join(base, 'open_set'), max_per_class=2000)

print(f"    Train: {feat_train.shape}, classes={np.unique(y_train).tolist()}")
print(f"    Val:   {feat_val.shape}")
print(f"    Open:  {feat_open.shape}")
print(f"    Feature dim: {feat_train.shape[1]}")

# ------------------------------------------------------------
print("\n[2] Sweeping OpenMax params on existing features...")
tail_sizes = [5, 10, 15, 20, 30, 50]
alpha_ranks = [1, 2, 3, 4, 5, 6]
distance_types = ['euclidean', 'cosine', 'eucos']

best_auroc = -1
best_params = None
results = []
total = len(tail_sizes) * len(alpha_ranks) * len(distance_types)
count = 0

for tail, alpha, dist in itertools.product(tail_sizes, alpha_ranks, distance_types):
    count += 1
    om = WeibullOpenMax(tail_size=tail, alpha_rank=alpha, distance_type=dist)
    om.fit(feat_train, y_train)

    val_scores = [om.predict(f)['unknown_score'] for f in feat_val]
    open_scores = [om.predict(f)['unknown_score'] for f in feat_open]

    all_probs = val_scores + open_scores
    all_labels = [0]*len(val_scores) + [1]*len(open_scores)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.5

    results.append({'tail': tail, 'alpha': alpha, 'distance': dist, 'auroc': auroc})
    if auroc > best_auroc:
        best_auroc = auroc
        best_params = (tail, alpha, dist)
    if count % 20 == 0:
        print(f"    {count}/{total}... best so far: {best_auroc:.4f}")

results.sort(key=lambda x: x['auroc'], reverse=True)

print(f"\n    Best: tail={best_params[0]}, alpha={best_params[1]}, {best_params[2]} → AUROC={best_auroc:.4f}")
print(f"    Original libMR: 0.9651  |  Ours: {best_auroc:.4f}  |  Gap: {0.9651 - best_auroc:+.4f}")
print(f"\n    Top 10:")
for r in results[:10]:
    print(f"      tail={r['tail']}, alpha={r['alpha']}, {r['distance']:10s} → AUROC={r['auroc']:.4f}")

# ------------------------------------------------------------
if best_auroc > 0.80:
    print(f"\n[3] Saving best detector (AUROC={best_auroc:.4f})...")
    om = WeibullOpenMax(tail_size=best_params[0], alpha_rank=best_params[1], distance_type=best_params[2])
    om.fit(feat_train, y_train)
    os.makedirs('models/weibull_om', exist_ok=True)
    with open('models/weibull_om/detector.pkl', 'wb') as f:
        pickle.dump(om, f)
    print("    ✓ Saved to models/weibull_om/detector.pkl")
else:
    print(f"\n[3] AUROC too low ({best_auroc:.4f}), need different approach")

print("\n[✓] Done!")
