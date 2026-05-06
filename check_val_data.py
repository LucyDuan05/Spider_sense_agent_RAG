import numpy as np
data = np.load('/home/mihu/CROSR-main/processed_cicids/val_known.npz', allow_pickle=True)
X, y = data['x'], data['y']
labels = data['label_names']
print('X shape:', X.shape)
print('Labels:', labels)
for i, name in enumerate(labels):
    idx = (y == i).sum()
    print(f'class {i} ({name}): {idx} samples')
