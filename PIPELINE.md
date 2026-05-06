# OpenMax IDS Pipeline

Full workflow from raw data to AUROC results. Run from `/home/mihu/CROSR-main`.

---

## Step 0: Compile libMR (first time only)

```bash
cd /home/mihu/CROSR-main/libMR
chmod +x compile.sh
./compile.sh
```

---

## Step 1: Data Preparation

Each dataset has a dedicated script. Run with `openmax` env.

```bash
conda activate openmax

# NSL-KDD
python prepare_nslkdd_1d.py

# CICIDS-2017
python prepare_cicids_1d.py

# CICIDS-2018
python prepare_cicids2018_1d.py

# UNSW-NB15
python prepare_unsw_nb15_1d.py
python make_unsw_val_split.py   # creates val_known.npz (20% stratified split)
```

Output: `processed_<dataset>/train_known.npz`, `val_known.npz`, `open_set.npz`

---

## Step 2: Train DHRNet-1D

```bash
conda activate openmax
python train_net_1d.py \
    --dataset <dataset_name> \
    --save_path save_models/<dataset_name>/best.pth
```

Checkpoint saved to `save_models/<dataset>_1d/best.pth`.

---

## Step 3: Extract Features

```bash
conda activate openmax
python get_model_features_1d.py \
    --train_path processed_<dataset>/train_known.npz \
    --val_path   processed_<dataset>/val_known.npz \
    --open_path  processed_<dataset>/open_set.npz \
    --save_path  saved_features/<dataset>_1d \
    --load_path  save_models/<dataset>_1d/best.pth
```

Output: `saved_features/<dataset>_1d/{train,val,open_set}/<class>/<sample>.npy`

---

## Step 4: Compute MAVs

```bash
conda activate openmax
python MAV_Compute.py \
    --feature_path saved_features/<dataset>_1d/train \
    --save_path    saved_MAVs/<dataset>_1d
```

---

## Step 5: Compute Distance Scores

```bash
conda activate openmax
python compute_distances.py \
    --feature_path saved_features/<dataset>_1d/train \
    --MAV_path     saved_MAVs/<dataset>_1d \
    --save_path    saved_distance_scores/<dataset>_1d
```

---

## Step 6: OpenMax Parameter Scan

Loads all features into RAM once, then sweeps 144 parameter combinations in memory.

```bash
PYTHONPATH=/home/mihu/CROSR-main/libMR:$PYTHONPATH \
conda run -n openmax_py2 python fast_scan.py \
    --MAV_path             saved_MAVs/<dataset>_1d \
    --distance_scores_path saved_distance_scores/<dataset>_1d \
    --feature_dir          saved_features/<dataset>_1d \
    --tail_sizes   5,10,15,20,30,50 \
    --alpha_ranks  1,2,3,4,5,6,8,10 \
    --distance_types eucos,euclidean,cosine \
    --top_k 5 | tee /tmp/scan_<dataset>.txt
```

---

## Step 7: Patch app.py with Real AUROC Values

```bash
conda activate openmax
python extract_best_auroc.py \
    --cicids     /tmp/scan_cicids.txt \
    --cicids2018 /tmp/scan_cicids2018.txt \
    --unsw       /tmp/scan_unsw.txt
```

Updates `web-frontend/backend/app.py` and saves `best_auroc_results.json`.

---

## Run Full Pipeline (All Datasets)

```bash
bash run_openmax_pipeline.sh
```

This script runs Steps 3–7 for all datasets (assumes models are already trained).

---

## Current Results

Metrics from `evaluate_openmax.py` at best balanced-accuracy threshold. precision/recall/f1 are for open-set (unknown) class detection.

| Dataset | AUROC | AUPR_OUT | FPR@95TPR | Precision | Recall | F1 | Bal.Acc | Best Params |
|---------|-------|----------|-----------|-----------|--------|----|---------|-------------|
| CICIDS-2017 | 0.9651 | 0.9482 | 0.1243 | 0.8937 | 0.9357 | 0.9143 | 0.9210 | tail=30, alpha=3, euclidean |
| UNSW-NB15 | 0.8949 | 0.6106 | 0.3627 | 0.5843 | 0.8718 | 0.6997 | 0.8396 | tail=50, alpha=6, cosine |
| NSL-KDD | 0.7587 | 0.2584 | 0.6873 | 0.2823 | 0.7835 | 0.4148 | 0.7417 | tail=50, alpha=1, euclidean |
| CICIDS-2018 | 0.4613 | 0.3446 | 0.7100 | 0.4665 | 0.9957 | 0.6344 | 0.6318 | tail=5, alpha=1, euclidean |
