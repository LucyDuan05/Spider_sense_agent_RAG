"""Split train_known.npz for UNSW-NB15 to create a val set (20% stratified)."""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = "./processed_unsw_nb15"
SEED = 42
VAL_RATIO = 0.2

data = np.load(f"{DATA_DIR}/train_known.npz", allow_pickle=True)
x, y, label_names = data["x"], data["y"], data["label_names"]

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
train_idx, val_idx = next(sss.split(x, y))

np.savez(f"{DATA_DIR}/val_known.npz", x=x[val_idx], y=y[val_idx], label_names=label_names)

unique, counts = np.unique(y[val_idx], return_counts=True)
print(f"Created val split: {len(val_idx)} samples")
for cls, cnt in zip(unique, counts):
    print(f"  class {cls} ({label_names[cls]}): {cnt}")
