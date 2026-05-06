import numpy as np
data = np.load('/home/mihu/CROSR-main/processed_cicids/open_set.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
for k in data.keys():
    v = data[k]
    print(f'  {k}: shape={v.shape if hasattr(v, "shape") else "scalar"}, dtype={v.dtype}')
    if hasattr(v, 'shape') and v.ndim == 1 and v.shape[0] < 20:
        print(f'    values: {v}')
