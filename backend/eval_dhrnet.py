"""
Test open-set detection using the ORIGINAL DHRNet-1D model features.
DHRNet was trained with reconstruction loss → better embedding separation.
"""
import numpy as np
import sys, os, pickle
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from DHR_Net_1D import DHRNet1D

from backend.detection_engine import AdaptiveThresholdDetector
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------
# 1. Load DHRNet model and extract features
# ------------------------------------------------------------
print("[1] Loading DHRNet-1D model...")
checkpoint = torch.load('save_models/cicids_1d/best.pth', map_location='cpu', weights_only=False)

state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
num_classes = checkpoint.get('num_classes', 6)
label_names = checkpoint.get('label_names', ['BENIGN','DDoS','DoS Hulk','PortScan','FTP-Patator','SSH-Patator'])
input_channels = checkpoint.get('input_channels', 1)
base_channels = checkpoint.get('base_channels', 128)
hidden_dim = checkpoint.get('hidden_dim', 512)

print(f"    classes={num_classes}, input_channels={input_channels}, base_channels={base_channels}, hidden_dim={hidden_dim}")

model = DHRNet1D(
    num_classes=num_classes,
    input_channels=input_channels,
    base_channels=base_channels,
    hidden_dim=hidden_dim,
)
model.load_state_dict(state_dict)
model.eval()
print(f"    ✓ Model loaded")

# Feature extractor: returns concatenation of [logits + pooled_latent_features]
# This mimics the original CROSR feature extraction approach
pool = nn.AdaptiveAvgPool1d(1)

def extract_dhr_features(model, x):
    """Extract CROSR-style features: [logits, pooled_z3, pooled_z2, pooled_z1]"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    if x.dim() == 2:
        x = x.unsqueeze(1)  # [batch, 1, features]

    with torch.no_grad():
        logits, reconstruct, latent = model(x)

    # Pool latent features (z3, z2, z1)
    pooled = [pool(z).flatten(start_dim=1) for z in latent]

    # Concatenate all
    features = torch.cat([logits] + pooled, dim=1)
    return features.numpy(), logits.numpy()

# ------------------------------------------------------------
# 2. Extract features for known + unknown
# ------------------------------------------------------------
print("\n[2] Extracting features...")
val = np.load('processed_cicids/val_known.npz', allow_pickle=True)
opn = np.load('processed_cicids/open_set.npz', allow_pickle=True)

def extract_batch(data_x, batch_size=256, desc=""):
    features_list = []
    logits_list = []
    for i in range(0, len(data_x), batch_size):
        batch = data_x[i:i+batch_size].astype(np.float32)
        feats, logits = extract_dhr_features(model, batch)
        features_list.append(feats)
        logits_list.append(logits)
        if (i // batch_size) % 20 == 0:
            print(f"    {desc} {i}/{len(data_x)}")
    return np.concatenate(features_list), np.concatenate(logits_list)

N = 2000
feat_k, logits_k = extract_batch(val['x'][:N].astype(np.float32), desc="known")
feat_u, logits_u = extract_batch(opn['x'][:N].astype(np.float32), desc="unknown")

print(f"    Known features:   {feat_k.shape}")
print(f"    Unknown features: {feat_u.shape}")

# ------------------------------------------------------------
# 3. Embedding space analysis
# ------------------------------------------------------------
print("\n[3] Embedding space analysis:")

# Norms
nk = np.linalg.norm(feat_k, axis=1)
nu = np.linalg.norm(feat_u, axis=1)
print(f"    Known norm:   mean={nk.mean():.2f} std={nk.std():.2f}")
print(f"    Unknown norm: mean={nu.mean():.2f} std={nu.std():.2f}")

# Max logit
max_logit_k = logits_k.max(axis=1)
max_logit_u = logits_u.max(axis=1)
print(f"    Known max_logit:   mean={max_logit_k.mean():.2f} std={max_logit_k.std():.2f}")
print(f"    Unknown max_logit: mean={max_logit_u.mean():.2f} std={max_logit_u.std():.2f}")

# Pairwise distances
kk_dist = []
ku_dist = []
step = 50
for i in range(0, N, step):
    for j in range(i+step, N, step):
        kk_dist.append(np.linalg.norm(feat_k[i] - feat_k[j]))
        break
    for j in range(0, N, step):
        ku_dist.append(np.linalg.norm(feat_k[i] - feat_u[j]))
        break

kk_dist = np.array(kk_dist)
ku_dist = np.array(ku_dist)
print(f"    Known-Known dist:   mean={kk_dist.mean():.3f} std={kk_dist.std():.3f}")
print(f"    Known-Unknown dist: mean={ku_dist.mean():.3f} std={ku_dist.std():.3f}")
print(f"    Separation: {ku_dist.mean() - kk_dist.mean():.3f}")

# ------------------------------------------------------------
# 4. Fit and evaluate open-set detector
# ------------------------------------------------------------
print("\n[4] Evaluating AdaptiveThreshold...")
y_train = val['y'][:N]

for sens in [1.0, 1.5, 2.0, 2.5, 3.0]:
    detector = AdaptiveThresholdDetector(sensitivity=sens)
    detector.fit(feat_k, y_train)

    # Count FP on known, TP on unknown
    fp = sum(1 for i in range(0, N, 10) if detector.predict(feat_k[i], return_details=True)['is_unknown'])
    tp = sum(1 for i in range(0, N, 10) if detector.predict(feat_u[i], return_details=True)['is_unknown'])
    print(f"    sens={sens}: known_fp={fp}/200 ({fp/2:.0f}%), unknown_tp={tp}/200 ({tp/2:.0f}%)")

# Compute full unknown_prob for AUROC
detector = AdaptiveThresholdDetector(sensitivity=2.0)
detector.fit(feat_k, y_train)

known_probs = []
for i in range(0, N, 10):
    r = detector.predict(feat_k[i], return_details=True)
    known_probs.append(r['openmax_prob'])

unknown_probs = []
for i in range(0, N, 10):
    r = detector.predict(feat_u[i], return_details=True)
    unknown_probs.append(r['openmax_prob'])

all_probs = known_probs + unknown_probs
all_labels = [0]*len(known_probs) + [1]*len(unknown_probs)

try:
    auroc = roc_auc_score(all_labels, all_probs)
    print(f"\n    🎯 AUROC: {auroc:.4f}")
    if auroc > 0.75:
        print("    ✅ Good! Using DHRNet features.")
    elif auroc > 0.65:
        print("    ⚠️  Moderate. Needs tuning.")
    else:
        print("    ❌ Poor. Need different approach.")
except Exception as e:
    print(f"    AUROC error: {e}")

# ------------------------------------------------------------
# 5. Save DHRNet-based openset detector if good
# ------------------------------------------------------------
if auroc > 0.65:
    print("\n[5] Saving DHRNet-based detector...")

    # Fit on full training data
    train = np.load('processed_cicids/train_known.npz', allow_pickle=True)
    x_train = train['x'].astype(np.float32)
    y_train_full = train['y']

    # Extract features for all training data (sample 10000 for speed)
    sample_idx = np.random.choice(len(x_train), min(10000, len(x_train)), replace=False)
    feat_train, _ = extract_batch(x_train[sample_idx])

    detector = AdaptiveThresholdDetector(sensitivity=2.0)
    detector.fit(feat_train, y_train_full[sample_idx])
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(feat_train)

    os.makedirs('models/dhrnet_detector', exist_ok=True)
    with open('models/dhrnet_detector/openset.pkl', 'wb') as f:
        pickle.dump({
            'centroids': detector.class_centroids,
            'thresholds': detector.class_thresholds,
            'global_threshold': detector.global_threshold,
            'iso_forest': iso,
        }, f)
    # Also save model config
    torch.save({
        'state_dict': state_dict,
        'num_classes': num_classes,
        'label_names': label_names,
        'input_channels': input_channels,
        'base_channels': base_channels,
        'hidden_dim': hidden_dim,
    }, 'models/dhrnet_detector/model.pth')
    print("    ✓ Saved to models/dhrnet_detector/")

print("\n[✓] Done!")
