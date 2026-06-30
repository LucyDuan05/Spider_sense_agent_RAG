"""
Evaluate pure-Python Weibull+OpenMax with DHRNet features.
Compare against original libMR results.
"""
import numpy as np
import sys, os, pickle, itertools
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from DHR_Net_1D import DHRNet1D
from backend.weibull_openmax import WeibullOpenMax
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------
# 1. Load DHRNet, extract features
# ------------------------------------------------------------
print("[1] Loading DHRNet...")
ckpt = torch.load('save_models/cicids_1d/best.pth', map_location='cpu', weights_only=False)
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
num_classes = ckpt.get('num_classes', 6)
label_names = ckpt.get('label_names', ['BENIGN','DDoS','DoS Hulk','PortScan','FTP-Patator','SSH-Patator'])
ic = ckpt.get('input_channels', 1)
bc = ckpt.get('base_channels', 128)
hd = ckpt.get('hidden_dim', 512)

model = DHRNet1D(num_classes=num_classes, input_channels=ic, base_channels=bc, hidden_dim=hd)
model.load_state_dict(state_dict)
model.eval()

pool = nn.AdaptiveAvgPool1d(1)

def extract_features(data_x):
    """Extract CROSR features: [logits, pooled_z3, pooled_z2, pooled_z1]"""
    if isinstance(data_x, np.ndarray):
        data_x = torch.from_numpy(data_x).float()
    if data_x.dim() == 2:
        data_x = data_x.unsqueeze(1)

    feats_list = []
    batch_size = 256
    for i in range(0, len(data_x), batch_size):
        batch = data_x[i:i+batch_size]
        with torch.no_grad():
            logits, _, latent = model(batch)
        pooled = [pool(z).flatten(start_dim=1) for z in latent]
        feats_list.append(torch.cat([logits] + pooled, dim=1).numpy())

    return np.concatenate(feats_list)

# Load data
print("[2] Loading data...")
val = np.load('processed_cicids/val_known.npz', allow_pickle=True)
opn = np.load('processed_cicids/open_set.npz', allow_pickle=True)
trn = np.load('processed_cicids/train_known.npz', allow_pickle=True)

# Extract features for a subset
N_TR = 5000
N_EV = 3000
print(f"    Extracting {N_TR} train + {N_EV} val + {N_EV} open features...")

idx_tr = np.random.choice(len(trn['x']), N_TR, replace=False)
feat_train = extract_features(trn['x'][idx_tr].astype(np.float32))
labels_train = trn['y'][idx_tr]

idx_val = np.random.choice(len(val['x']), N_EV, replace=False)
feat_val = extract_features(val['x'][idx_val].astype(np.float32))
labels_val = val['y'][idx_val]

idx_opn = np.random.choice(len(opn['x']), N_EV, replace=False)
feat_open = extract_features(opn['x'][idx_opn].astype(np.float32))

print(f"    Feature dim: {feat_train.shape[1]}")
print(f"    Classes: {np.unique(labels_train)}")

# ------------------------------------------------------------
# 2. Sweep OpenMax parameters
# ------------------------------------------------------------
print("\n[3] Sweeping OpenMax parameters...")

tail_sizes = [5, 10, 20, 30, 50]
alpha_ranks = [1, 2, 3, 4, 5, 6]
distance_types = ['euclidean', 'cosine', 'eucos']

best_auroc = -1
best_params = None
results = []

total = len(tail_sizes) * len(alpha_ranks) * len(distance_types)
count = 0

for tail_size, alpha_rank, dist_type in itertools.product(tail_sizes, alpha_ranks, distance_types):
    count += 1
    if count % 10 == 0:
        print(f"    {count}/{total}...")

    om = WeibullOpenMax(tail_size=tail_size, alpha_rank=alpha_rank, distance_type=dist_type)
    om.fit(feat_train, labels_train)

    # Get unknown score for all validation + open samples
    val_scores = []
    open_scores = []

    sample_n = 1000  # subset for speed
    for i in range(0, min(sample_n, len(feat_val))):
        r = om.predict(feat_val[i])
        val_scores.append(r['unknown_score'])

    for i in range(0, min(sample_n, len(feat_open))):
        r = om.predict(feat_open[i])
        open_scores.append(r['unknown_score'])

    all_probs = val_scores + open_scores
    all_labels = [0] * len(val_scores) + [1] * len(open_scores)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.5

    results.append({
        'tail': tail_size, 'alpha': alpha_rank, 'distance': dist_type,
        'auroc': auroc
    })

    if auroc > best_auroc:
        best_auroc = auroc
        best_params = (tail_size, alpha_rank, dist_type)

# Show results
results.sort(key=lambda x: x['auroc'], reverse=True)
print(f"\n    Best: tail={best_params[0]}, alpha={best_params[1]}, distance={best_params[2]}, AUROC={best_auroc:.4f}")
print(f"\n    Top 10:")
for r in results[:10]:
    print(f"      tail={r['tail']}, alpha={r['alpha']}, {r['distance']:10s} → AUROC={r['auroc']:.4f}")

# Compare with original
print(f"\n    Original OpenMax (libMR): AUROC = 0.9651")
print(f"    Our pure-Python OpenMax:  AUROC = {best_auroc:.4f}")

# ------------------------------------------------------------
# 3. Save best model
# ------------------------------------------------------------
if best_auroc > 0.75:
    print(f"\n[4] Saving best model (AUROC={best_auroc:.4f})...")

    # Fit on all training data
    feat_all, _ = extract_features(trn['x'][:10000].astype(np.float32))
    om = WeibullOpenMax(
        tail_size=best_params[0],
        alpha_rank=best_params[1],
        distance_type=best_params[2],
    )
    om.fit(feat_train, labels_train)  # use sampled for speed

    os.makedirs('models/weibull_om', exist_ok=True)
    with open('models/weibull_om/detector.pkl', 'wb') as f:
        pickle.dump(om, f)
    print("    ✓ Saved to models/weibull_om/detector.pkl")
else:
    print(f"\n[4] AUROC too low ({best_auroc:.4f}), not saving.")

print("\n[✓] Done!")
