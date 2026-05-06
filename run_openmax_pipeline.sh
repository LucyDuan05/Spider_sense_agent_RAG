#!/bin/bash
# Full OpenMax pipeline: feature extraction + fast parameter scan.
# NSL-KDD AUROC is already known: 0.9648
# Run from: /home/mihu/CROSR-main

set -e
cd /home/mihu/CROSR-main
source /home/mihu/miniconda3/etc/profile.d/conda.sh

ROOT=/home/mihu/CROSR-main
TAILS="5,10,15,20,30,50"
ALPHAS="1,2,3,4,5,6,8,10"
DISTS="eucos,euclidean,cosine"

fast_scan() {
    local name=$1 mav=$2 dist_sc=$3 feat=$4 out=$5
    echo "--- Fast-scanning $name ---"
    conda run -n openmax_py2 bash -c \
        "PYTHONPATH=$ROOT/libMR:\$PYTHONPATH python $ROOT/fast_scan.py \
            --MAV_path $mav \
            --distance_scores_path $dist_sc \
            --feature_dir $feat \
            --tail_sizes $TAILS \
            --alpha_ranks $ALPHAS \
            --distance_types $DISTS \
            --top_k 5" 2>&1 | tee "$out"
    echo ""
}

echo "================================================================"
echo "STEP 1: CICIDS-2017 — fast scan (all artifacts exist)"
echo "================================================================"
fast_scan cicids_1d \
    $ROOT/saved_MAVs/cicids_1d \
    $ROOT/saved_distance_scores/cicids_1d \
    $ROOT/saved_features/cicids_1d \
    /tmp/scan_cicids.txt

echo "================================================================"
echo "STEP 2: CICIDS-2018 v2 — extract features"
echo "================================================================"
conda run -n openmax python $ROOT/get_model_features_1d.py \
    --train_path $ROOT/processed_cicids2018/train_known.npz \
    --val_path   $ROOT/processed_cicids2018/val_known.npz \
    --open_path  $ROOT/processed_cicids2018/open_set.npz \
    --save_path  $ROOT/saved_features/cicids2018_1d_v2 \
    --load_path  $ROOT/save_models/cicids2018_1d_v2/best.pth
echo ""

echo "================================================================"
echo "STEP 3: CICIDS-2018 v2 — fast scan"
echo "================================================================"
fast_scan cicids2018_1d_v2 \
    $ROOT/saved_MAVs/cicids2018_1d_v2 \
    $ROOT/saved_distance_scores/cicids2018_1d_v2 \
    $ROOT/saved_features/cicids2018_1d_v2 \
    /tmp/scan_cicids2018.txt

echo "================================================================"
echo "STEP 4: UNSW-NB15 — create val split from train data"
echo "================================================================"
conda run -n openmax python $ROOT/make_unsw_val_split.py
echo ""

echo "================================================================"
echo "STEP 5: UNSW-NB15 — extract features"
echo "================================================================"
conda run -n openmax python $ROOT/get_model_features_1d.py \
    --train_path $ROOT/processed_unsw_nb15/train_known.npz \
    --val_path   $ROOT/processed_unsw_nb15/val_known.npz \
    --open_path  $ROOT/processed_unsw_nb15/open_set.npz \
    --save_path  $ROOT/saved_features/unsw_nb15_1d \
    --load_path  $ROOT/save_models/unsw_nb15_1d/best.pth
echo ""

echo "================================================================"
echo "STEP 6: UNSW-NB15 — fast scan"
echo "================================================================"
fast_scan unsw_nb15_1d \
    $ROOT/saved_MAVs/unsw_nb15_1d \
    $ROOT/saved_distance_scores/unsw_nb15_1d \
    $ROOT/saved_features/unsw_nb15_1d \
    /tmp/scan_unsw.txt

echo "================================================================"
echo "EXTRACTING BEST RESULTS + PATCHING app.py"
echo "================================================================"
conda run -n openmax python $ROOT/extract_best_auroc.py \
    --cicids     /tmp/scan_cicids.txt \
    --cicids2018 /tmp/scan_cicids2018.txt \
    --unsw       /tmp/scan_unsw.txt

echo ""
echo "All done. web-frontend/backend/app.py has been updated."
