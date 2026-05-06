#!/bin/bash
# Post-training pipeline for one dataset:
# features → MAV → distances → fast_scan → evaluate_openmax
# Usage: bash run_postprocess.sh <dataset_name> <feature_dim> <n_classes>
# Example: bash run_postprocess.sh cicids_1d 78 6

set -e
source /home/mihu/miniconda3/etc/profile.d/conda.sh
cd /home/mihu/CROSR-main

DATASET=$1   # e.g. cicids_1d
DATA_DIR=$2  # e.g. processed_cicids

ROOT=/home/mihu/CROSR-main
TAILS="5,10,15,20,30,50"
ALPHAS="1,2,3,4,5,6,8,10"
DISTS="eucos,euclidean,cosine"

echo "=== [$DATASET] Step 1: Extract features ==="
conda run -n openmax python $ROOT/get_model_features_1d.py \
    --train_path $ROOT/$DATA_DIR/train_known.npz \
    --val_path   $ROOT/$DATA_DIR/val_known.npz \
    --open_path  $ROOT/$DATA_DIR/open_set.npz \
    --save_path  $ROOT/saved_features/$DATASET \
    --load_path  $ROOT/save_models/$DATASET/best.pth

echo "=== [$DATASET] Step 2: Compute MAVs ==="
conda run -n openmax python $ROOT/MAV_Compute.py \
    --feature_path $ROOT/saved_features/$DATASET/train \
    --save_path    $ROOT/saved_MAVs/$DATASET

echo "=== [$DATASET] Step 3: Compute distances ==="
conda run -n openmax python $ROOT/compute_distances.py \
    --feature_path $ROOT/saved_features/$DATASET/train \
    --MAV_path     $ROOT/saved_MAVs/$DATASET \
    --save_path    $ROOT/saved_distance_scores/$DATASET

echo "=== [$DATASET] Step 4: Fast scan ==="
conda run -n openmax_py2 bash -c \
    "PYTHONPATH=$ROOT/libMR:\$PYTHONPATH python $ROOT/fast_scan.py \
        --MAV_path             $ROOT/saved_MAVs/$DATASET \
        --distance_scores_path $ROOT/saved_distance_scores/$DATASET \
        --feature_dir          $ROOT/saved_features/$DATASET \
        --tail_sizes $TAILS --alpha_ranks $ALPHAS \
        --distance_types $DISTS --top_k 5" \
    2>&1 | tee /tmp/scan_${DATASET}.txt

echo "=== [$DATASET] Done ==="
