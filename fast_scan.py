# -*- coding: utf-8 -*-
"""
Fast OpenMax parameter scan.

Loads all val/open_set features into RAM once, then sweeps parameter
combinations entirely in memory -- avoiding 144x repeated disk I/O.

Compatible with Python 2 (openmax_py2 env) and Python 3.
"""
from __future__ import print_function

import argparse
import itertools
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from evt_fitting import weibull_tailfitting, query_weibull
from evaluate_openmax import (
    compute_threshold_metrics,
    find_best_balanced_threshold,
    find_fpr_at_target_tpr,
)


# ---------------------------------------------------------------------------
# feature loading
# ---------------------------------------------------------------------------

def load_split(feature_dir, split):
    """Load all feature vectors for *split* into a single (N, D) array."""
    split_dir = os.path.join(feature_dir, split)
    rows = []
    for cls_name in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            feat = np.load(os.path.join(cls_dir, fname))   # shape (1, D)
            rows.append(feat[0])
    if not rows:
        raise RuntimeError("No features found in {}".format(split_dir))
    arr = np.array(rows, dtype=np.float64)
    print("  loaded {} x {} from {}/{}".format(arr.shape[0], arr.shape[1],
                                                os.path.basename(feature_dir),
                                                split))
    return arr


# ---------------------------------------------------------------------------
# per-sample OpenMax probability (vectorised over classes, scalar per sample)
# ---------------------------------------------------------------------------

def openmax_prob(logits, mav_list, weibull_list, alpha_rank, distance_type):
    """
    Compute the OpenMax unknown-class probability for one feature vector.

    Parameters
    ----------
    logits       : (D,)  full feature vector; first NCLASSES entries are class logits
    mav_list     : list of (D,) mean activation vectors, one per class
    weibull_list : list of fitted libmr.MR objects, one per class
    alpha_rank   : int
    distance_type: str  'euclidean' | 'cosine' | 'eucos'
    """
    import scipy.spatial.distance as spd

    NCLASSES = len(mav_list)
    alpharank = min(alpha_rank, NCLASSES)

    # --- alpha weights ---
    ranked_idx = logits[:NCLASSES].argsort()[::-1]
    ranked_alpha = np.zeros(NCLASSES)
    for j in range(alpharank):
        ranked_alpha[ranked_idx[j]] = (alpharank + 1 - (j + 1)) / float(alpharank)

    # --- Weibull correction ---
    openmax_fc8 = np.empty(NCLASSES)
    openmax_unk = np.empty(NCLASSES)
    for c in range(NCLASSES):
        if distance_type == 'euclidean':
            dist = spd.euclidean(mav_list[c], logits)
        elif distance_type == 'cosine':
            dist = spd.cosine(mav_list[c], logits)
        else:   # eucos
            dist = spd.euclidean(mav_list[c], logits) / 200.0 + spd.cosine(mav_list[c], logits)
        wscore = weibull_list[c].w_score(dist)
        mod = logits[c] * (1.0 - wscore * ranked_alpha[c])
        openmax_fc8[c] = mod
        openmax_unk[c] = logits[c] - mod

    unknown_logit = float(np.sum(openmax_unk))
    all_logits = np.append(openmax_fc8, unknown_logit)
    max_l = np.max(all_logits)
    exp_l = np.exp(all_logits - max_l)
    total = np.sum(exp_l)
    if total == 0.0 or not np.isfinite(total):
        return 0.0
    return float(exp_l[-1] / total)


# ---------------------------------------------------------------------------
# batch scoring (pre-loaded features, per tail+dist combo)
# ---------------------------------------------------------------------------

def score_batch(features, mav_list, weibull_list, alpha_rank, distance_type):
    return np.array([
        openmax_prob(f, mav_list, weibull_list, alpha_rank, distance_type)
        for f in features
    ])


# ---------------------------------------------------------------------------
# main scan
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='Fast OpenMax parameter scan (features loaded once)')
    p.add_argument('--MAV_path',             required=True)
    p.add_argument('--distance_scores_path', required=True)
    p.add_argument('--feature_dir',          required=True)
    p.add_argument('--tail_sizes',   default='5,10,15,20,30,50')
    p.add_argument('--alpha_ranks',  default='1,2,3,4,5,6,8,10')
    p.add_argument('--distance_types', default='eucos,euclidean,cosine')
    p.add_argument('--top_k', default=5, type=int)
    return p.parse_args()


def main():
    args = get_args()

    tail_sizes    = [int(x) for x in args.tail_sizes.split(',')    if x.strip()]
    alpha_ranks   = [int(x) for x in args.alpha_ranks.split(',')   if x.strip()]
    distance_types = [x.strip() for x in args.distance_types.split(',') if x.strip()]

    # ---- load features once ----
    print("Loading features...")
    sys.stdout.flush()
    val_feats  = load_split(args.feature_dir, 'val')
    open_feats = load_split(args.feature_dir, 'open_set')
    labels = np.array([0] * len(val_feats) + [1] * len(open_feats))
    print("  total: {} val + {} open_set = {} samples".format(
        len(val_feats), len(open_feats), len(labels)))
    sys.stdout.flush()

    results = []

    for distance_type, tail_size in itertools.product(distance_types, tail_sizes):
        # Fit Weibull model once per (dist_type, tail_size)
        weibull_model = weibull_tailfitting(
            args.MAV_path,
            args.distance_scores_path,
            tailsize=tail_size,
            distance_type=distance_type,
        )
        NCLASSES = len(weibull_model)

        # Extract MAV vectors and Weibull objects as ordered lists
        mav_list     = [query_weibull(str(c), weibull_model, distance_type)[0] for c in range(NCLASSES)]
        weibull_list = [query_weibull(str(c), weibull_model, distance_type)[2] for c in range(NCLASSES)]

        for alpha_rank in alpha_ranks:
            val_scores  = score_batch(val_feats,  mav_list, weibull_list, alpha_rank, distance_type)
            open_scores = score_batch(open_feats, mav_list, weibull_list, alpha_rank, distance_type)

            scores = np.concatenate([val_scores, open_scores])
            auroc  = roc_auc_score(labels, scores)
            aupr   = average_precision_score(labels, scores)
            fpr95, _ = find_fpr_at_target_tpr(labels, scores, 0.95)
            best   = find_best_balanced_threshold(labels, scores)
            bal_acc = best['balanced_accuracy']

            rec = dict(tail_size=tail_size, alpha_rank=alpha_rank,
                       distance_type=distance_type,
                       auroc=auroc, aupr=aupr,
                       fpr95=fpr95, bal_acc=bal_acc)
            results.append(rec)

            fpr95_s = '{:.6f}'.format(fpr95) if fpr95 is not None else 'n/a'
            print('tail={} alpha={} dist={} AUROC={:.6f} AUPR={:.6f} FPR95={} BAL_ACC={:.6f}'.format(
                tail_size, alpha_rank, distance_type, auroc, aupr, fpr95_s, bal_acc))
            sys.stdout.flush()

    ranked = sorted(results, key=lambda r: (r['auroc'], r['aupr'], r['bal_acc']), reverse=True)

    print('')
    print('Top {} results by AUROC:'.format(min(args.top_k, len(ranked))))
    for i, r in enumerate(ranked[:args.top_k], 1):
        fpr95_s = '{:.6f}'.format(r['fpr95']) if r['fpr95'] is not None else 'n/a'
        print('{0}. tail={1} alpha={2} dist={3} '
              'AUROC={4:.6f} AUPR={5:.6f} FPR95={6} BAL_ACC={7:.6f}'.format(
                  i, r['tail_size'], r['alpha_rank'], r['distance_type'],
                  r['auroc'], r['aupr'], fpr95_s, r['bal_acc']))


if __name__ == '__main__':
    main()
