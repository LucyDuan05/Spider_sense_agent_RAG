# CROSR Project Context

## Overview

CROSR (Classification-Reconstruction Learning for Open-Set Recognition) applied to Network Intrusion Detection Systems (IDS). The model is DHRNet (Dual-Head Reconstruction Network), adapted for 1D tabular network traffic features.

**Goal**: Detect both known attack classes and unknown/novel attacks (open-set recognition) using OpenMax.

---

## Environments

| Conda Env | Python | Purpose |
|-----------|--------|---------|
| `openmax` | 3.9 | Model training, feature extraction, data preparation |
| `openmax_py2` | 2.7 | OpenMax Weibull fitting (`libMR` requires Python 2) |

**PYTHONPATH required for `openmax_py2`:**
```bash
PYTHONPATH=/home/mihu/CROSR-main/libMR:$PYTHONPATH
```

---

## Datasets

| Dataset | Classes (known) | Open-set source | Status |
|---------|----------------|-----------------|--------|
| NSL-KDD | 5 | DoS/Probe/R2L/U2R split | Done |
| CICIDS-2017 | 6 | Held-out attack types | Done |
| CICIDS-2018 | 8 | Held-out attack types | Done |
| UNSW-NB15 | 6 | Held-out attack types | Done |

Processed data lives in:
- `/home/mihu/CROSR-main/processed_nslkdd/`
- `/home/mihu/CROSR-main/processed_cicids/`
- `/home/mihu/CROSR-main/processed_cicids2018/`
- `/home/mihu/CROSR-main/processed_unsw_nb15/`

Each directory contains: `train_known.npz`, `val_known.npz`, `open_set.npz`

> **Note**: UNSW-NB15 originally had no `val_known.npz`. It was created by `make_unsw_val_split.py` (20% stratified split from `train_known.npz`).

---

## OpenMax Results (Best Parameters)

All metrics from `evaluate_openmax.py` at the best balanced-accuracy threshold.
precision/recall/f1 are for the unknown (open-set) class detection.

| Dataset | AUROC | AUPR_OUT | AUPR_IN | FPR@95TPR | Precision | Recall | Bal.Acc | Best Params |
|---------|-------|----------|---------|-----------|-----------|--------|---------|-------------|
| CICIDS-2017 | 0.9651 | 0.9482 | 0.9759 | 0.1243 | 0.8937 | 0.9357 | 0.9210 | tail=30, alpha=3, euclidean |
| UNSW-NB15 | 0.8949 | 0.6106 | 0.9682 | 0.3627 | 0.5843 | 0.8718 | 0.8396 | tail=50, alpha=6, cosine |
| NSL-KDD | 0.7587 | 0.2584 | 0.9561 | 0.6873 | 0.2823 | 0.7835 | 0.7417 | tail=50, alpha=1, euclidean |
| CICIDS-2018 | 0.4613 | 0.3446 | 0.7132 | 0.7100 | 0.4665 | 0.9957 | 0.6318 | tail=5, alpha=1, euclidean |

> **Warning**: CICIDS-2018 AUROC (0.46) is near-random — the model cannot distinguish known from unknown traffic. Needs retraining or open-set split review.
> **Note**: NSL-KDD AUROC (0.76) is lower than expected. The previously reported 0.9648 was a CICIDS-2017 result mislabeled as NSL-KDD.

Results saved to: `best_auroc_results.json`

---

## Key Scripts

| Script | Env | Purpose |
|--------|-----|---------|
| `prepare_*_1d.py` | `openmax` | Preprocess raw CSV → `.npz` splits |
| `train_net_1d.py` | `openmax` | Train DHRNet-1D |
| `get_model_features_1d.py` | `openmax` | Extract penultimate-layer features |
| `MAV_Compute.py` | `openmax` | Compute Mean Activation Vectors per class |
| `compute_distances.py` | `openmax` | Compute Weibull fitting distances |
| `fast_scan.py` | `openmax_py2` | Sweep OpenMax params in-memory (fast) |
| `extract_best_auroc.py` | `openmax` | Parse scan results → patch `app.py` |
| `run_openmax_pipeline.sh` | both | Full pipeline orchestration |
| `make_unsw_val_split.py` | `openmax` | Create val split for UNSW-NB15 |

---

## Artifact Directories

```
saved_features/        # Per-sample .npy feature vectors (val/ + open_set/)
saved_MAVs/            # Mean Activation Vectors per class
saved_distance_scores/ # Weibull fitting distance scores
save_models/           # Trained model checkpoints (best.pth)
```

Each subdirectory is named after the dataset variant, e.g. `cicids_1d`, `cicids2018_1d_v2`, `nslkdd_1d`, `unsw_nb15_1d`.

---

## Web Frontend

- **Backend**: Flask (`web-frontend/backend/app.py`), port 5000
- **Frontend**: React + TypeScript, port 3000
- AUROC values in `app.py` → `DATASET_RESULTS` dict, patched automatically by `extract_best_auroc.py`
- Startup guide: `web-frontend/STARTUP.md`

---

## Known Issues / TODOs

- CICIDS-2018 AUROC is very low (0.46) — model `cicids2018_1d_v2` needs investigation
- `fast_scan.py` top-5 display had a Python 2/3 `format()` keyword conflict (fixed: use positional `{0}` format)
- `libMR` must be compiled before first use: `cd libMR && ./compile.sh`
