"""
Integrate WeibullOpenMax into the DetectionEngine and test full pipeline.
"""
import numpy as np
import sys, os, pickle, torch, torch.nn as nn
sys.path.insert(0, '.')

from DHR_Net_1D import DHRNet1D
from backend.weibull_openmax import WeibullOpenMax
from backend.detection_engine import DetectionEngine

# ------------------------------------------------------------
# 1. Load DHRNet model for feature extraction
# ------------------------------------------------------------
print("[1] Loading DHRNet model...")
ckpt = torch.load('save_models/cicids_1d/best.pth', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
model = DHRNet1D(
    num_classes=ckpt.get('num_classes', 6),
    input_channels=ckpt.get('input_channels', 1),
    base_channels=ckpt.get('base_channels', 128),
    hidden_dim=ckpt.get('hidden_dim', 512),
)
model.load_state_dict(sd)
model.eval()
pool = nn.AdaptiveAvgPool1d(1)

# ------------------------------------------------------------
# 2. Load WeibullOpenMax detector
# ------------------------------------------------------------
print("[2] Loading WeibullOpenMax detector...")
with open('models/weibull_om/detector.pkl', 'rb') as f:
    om = pickle.load(f)
print(f"    tail_size={om.tail_size}, alpha_rank={om.alpha_rank}, dist={om.distance_type}")
print(f"    {len(om.class_centroids)} classes: {list(om.class_centroids.keys())}")

# ------------------------------------------------------------
# 3. Test on validation + open-set
# ------------------------------------------------------------
print("\n[3] Testing...")
val = np.load('processed_cicids/val_known.npz', allow_pickle=True)
opn = np.load('processed_cicids/open_set.npz', allow_pickle=True)
label_names = val['label_names']

def predict_one(raw_features):
    """Full CROSR-OpenMax prediction on one sample."""
    x = torch.from_numpy(raw_features.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits, _, latent = model(x)
    pooled = [pool(z).flatten(start_dim=1) for z in latent]
    feature = torch.cat([logits] + pooled, dim=1).numpy().flatten()

    result = om.predict(feature)
    # Map prediction to label name
    cls_id = result['predicted_class']
    if result['is_unknown']:
        label = 'UNKNOWN'
    else:
        label = str(label_names[cls_id]) if cls_id < len(label_names) else f'class_{cls_id}'

    return label, result

# Known samples
print("    Known samples:")
for i in [0, 100, 500]:
    label, r = predict_one(val['x'][i])
    print(f"      [{i}] true={label_names[val['y'][i]]} → pred={label}, "
          f"unknown_score={r['unknown_score']:.3f}, is_unknown={r['is_unknown']}")

# Unknown samples
print("    Unknown samples:")
for i in [0, 10, 50]:
    label, r = predict_one(opn['x'][i])
    print(f"      [{i}] → pred={label}, "
          f"unknown_score={r['unknown_score']:.3f}, is_unknown={r['is_unknown']}")

# Quick AUROC
from sklearn.metrics import roc_auc_score
scores_k, scores_u = [], []
for i in range(0, 1000, 2):
    _, r = predict_one(val['x'][i])
    scores_k.append(r['unknown_score'])
for i in range(0, 1000, 2):
    _, r = predict_one(opn['x'][i])
    scores_u.append(r['unknown_score'])

all_scores = scores_k + scores_u
all_labels = [0]*len(scores_k) + [1]*len(scores_u)
auroc = roc_auc_score(all_labels, all_scores)
print(f"\n    Quick AUROC: {auroc:.4f}")

# ------------------------------------------------------------
# 4. Save integration checkpoint
# ------------------------------------------------------------
print("\n[4] Saving integration...")
os.makedirs('models/weibull_om', exist_ok=True)
torch.save({
    'model_state_dict': sd,
    'num_classes': ckpt.get('num_classes', 6),
    'label_names': label_names,
    'input_channels': ckpt.get('input_channels', 1),
    'base_channels': ckpt.get('base_channels', 128),
    'hidden_dim': ckpt.get('hidden_dim', 512),
}, 'models/weibull_om/model.pth')
# Also save om detector separately (already saved)
print("    ✓ model.pth and detector.pkl saved to models/weibull_om/")

print("\n[✓] Done! Integration ready.")
print("    Model: DHRNet-1D (reconstruction loss) from save_models/cicids_1d/")
print("    OpenSet: WeibullOpenMax (tail=50, alpha=4, euclidean)")
print(f"    AUROC: {auroc:.4f}")
