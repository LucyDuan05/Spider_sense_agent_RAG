"""
Pure Python 3 implementation of Weibull fitting + OpenMax.
Replaces libMR (Python 2.7) dependency.

Based on: "Toward Open Set Recognition" (Bendale & Boult, TPAMI 2016)
          "Classification-Reconstruction Learning for Open-Set Recognition" (Yoshihashi et al., CVPR 2019)
"""
import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import curve_fit


def _weibull_tail_cdf(x, scale, shape):
    """Weibull CDF for tail fitting: P(X <= x)."""
    return weibull_min.cdf(x, shape, scale=scale)


def fit_weibull_tail(distances, tail_size=20):
    """
    Fit Weibull distribution to the TAIL of distance distribution.
    Uses method-of-moments for robustness.

    Returns: (shape, scale) parameters
    """
    distances = np.sort(distances)

    if tail_size >= len(distances):
        tail_size = max(5, len(distances) // 4)

    tail = distances[-tail_size:]
    tail_mean = np.mean(tail)
    tail_var = np.var(tail)

    if tail_var < 1e-10 or tail_mean < 1e-10:
        return 1.5, max(tail_mean, 1e-6)

    # Method of moments for Weibull:
    # mean = scale * gamma(1 + 1/shape)
    # var  = scale^2 * [gamma(1 + 2/shape) - gamma(1 + 1/shape)^2]
    # Approximate: shape ≈ (mean / std)^1.086
    from scipy.special import gamma as gamma_func

    cv = np.sqrt(tail_var) / tail_mean  # coefficient of variation

    # Approximate shape from CV (empirical relationship for Weibull)
    if cv < 0.01:
        shape = 10.0  # nearly deterministic
    elif cv > 10.0:
        shape = 0.5
    else:
        shape = cv ** (-1.086)
        shape = max(0.5, min(20.0, shape))

    # Scale from mean
    scale = tail_mean / gamma_func(1.0 + 1.0 / shape)
    scale = max(1e-10, scale)

    if not np.isfinite(shape) or not np.isfinite(scale):
        return 1.5, tail_mean

    return float(shape), float(scale)


def weibull_cdf_probability(distance, shape, scale):
    """
    Survival function: P(X > distance) under the Weibull model.
    Lower value → distance is in the extreme tail → more likely unknown.
    """
    if scale <= 0 or not np.isfinite(scale):
        return 0.5
    if shape <= 0 or not np.isfinite(shape):
        return 0.5

    x = distance / scale
    if x < 0:
        x = 0.0

    try:
        # Survival function: exp(-(x)^shape)
        sf = np.exp(-(x ** shape))
        return float(np.clip(sf, 0.0, 1.0))
    except (OverflowError, FloatingPointError):
        if x > 1.0 and shape > 1.0:
            return 0.0  # very far in tail
        return 0.5


def compute_openmax_scores(features, class_centroids, weibull_params,
                           alpha_rank=3, distance_type='eucos'):
    """
    Compute OpenMax scores for a sample.

    Args:
        features: [D] feature vector for one sample
        class_centroids: dict {class_id: centroid_array}
        weibull_params: dict {class_id: (shape, scale)}
        alpha_rank: number of top classes to "recalibrate"
        distance_type: 'euclidean', 'cosine', or 'eucos'

    Returns:
        dict with openmax_scores, predicted_class, is_unknown, unknown_prob
    """
    features = np.asarray(features).flatten()
    num_classes = len(class_centroids)

    # Compute distances to each class centroid
    distances = {}
    for cls_id, centroid in class_centroids.items():
        centroid = np.asarray(centroid).flatten()
        if distance_type == 'euclidean':
            d = np.linalg.norm(features - centroid)
        elif distance_type == 'cosine':
            cos_sim = np.dot(features, centroid) / (np.linalg.norm(features) * np.linalg.norm(centroid) + 1e-10)
            d = 1.0 - cos_sim
        elif distance_type == 'eucos':
            eu = np.linalg.norm(features - centroid)
            cos_sim = np.dot(features, centroid) / (np.linalg.norm(features) * np.linalg.norm(centroid) + 1e-10)
            d = eu * (1.0 - cos_sim + 0.5)
        else:
            d = np.linalg.norm(features - centroid)
        distances[cls_id] = d

    # Compute raw "scores" as inverse of distance (higher = closer)
    raw_scores = {}
    for cls_id, d in distances.items():
        raw_scores[cls_id] = 1.0 / (d + 1e-6)

    # Normalize to [0, 1] like softmax
    total = sum(raw_scores.values())
    if total > 0:
        scores = {k: v / total for k, v in raw_scores.items()}
    else:
        scores = {k: 1.0 / num_classes for k in raw_scores}

    # Get Weibull probabilities for each class
    weibull_probs = {}
    for cls_id in class_centroids:
        shape, scale = weibull_params.get(cls_id, (1.5, 1.0))
        weibull_probs[cls_id] = weibull_cdf_probability(distances[cls_id], shape, scale)

    # Sort classes by score (highest first)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # OpenMax recalibration: modify top `alpha_rank` scores using Weibull
    openmax_scores = dict(scores)
    for rank_idx, (cls_id, score) in enumerate(ranked[:alpha_rank]):
        w = weibull_probs[cls_id]
        # Reduce score for classes with low Weibull probability
        openmax_scores[cls_id] = score * w

    # The "unknown" class score
    unknown_score = 1.0 - sum(openmax_scores.values())
    if unknown_score < 0:
        unknown_score = 0.0

    # Renormalize
    total = sum(openmax_scores.values()) + unknown_score
    for k in openmax_scores:
        openmax_scores[k] /= total
    unknown_score /= total

    # Prediction
    best_class = max(openmax_scores, key=openmax_scores.get)
    best_score = openmax_scores[best_class]

    # Is unknown if unknown_score is highest
    is_unknown = unknown_score > best_score

    return {
        'openmax_scores': {int(k): float(v) for k, v in openmax_scores.items()},
        'unknown_score': float(unknown_score),
        'predicted_class': int(best_class if not is_unknown else -1),
        'is_unknown': bool(is_unknown),
        'class_confidence': float(best_score),
        'all_distances': {int(k): float(v) for k, v in distances.items()},
        'weibull_probs': {int(k): float(v) for k, v in weibull_probs.items()},
    }


# ------------------------------------------------------------
# Full pipeline: fit Weibull models per class + evaluate
# ------------------------------------------------------------

class WeibullOpenMax:
    """Full Weibull + OpenMax pipeline (replaces libMR)."""

    def __init__(self, tail_size=20, alpha_rank=3, distance_type='eucos'):
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank
        self.distance_type = distance_type
        self.class_centroids = {}
        self.weibull_params = {}
        self.fitted = False

    def fit(self, features, labels):
        """
        Fit per-class centroids and Weibull tail models.

        Args:
            features: [N, D] feature vectors
            labels: [N] integer labels
        """
        features = np.asarray(features)
        labels = np.asarray(labels)
        unique_classes = np.unique(labels)

        for cls_id in unique_classes:
            mask = labels == cls_id
            cls_features = features[mask]

            # Centroid (MAV)
            centroid = cls_features.mean(axis=0).flatten()
            self.class_centroids[int(cls_id)] = centroid

            # Distances to centroid for all training samples of this class
            distances = np.array([
                self._compute_distance(f, centroid)
                for f in cls_features
            ])

            # Fit Weibull to tail
            shape, scale = fit_weibull_tail(distances, self.tail_size)
            self.weibull_params[int(cls_id)] = (shape, scale)

        self.fitted = True
        return self

    def _compute_distance(self, features, centroid):
        """Compute distance between feature and centroid."""
        f = np.asarray(features).flatten()
        c = np.asarray(centroid).flatten()

        eu = np.linalg.norm(f - c)

        if self.distance_type == 'euclidean':
            return eu
        elif self.distance_type == 'cosine':
            cos_sim = np.dot(f, c) / (np.linalg.norm(f) * np.linalg.norm(c) + 1e-10)
            return 1.0 - cos_sim
        elif self.distance_type == 'eucos':
            cos_sim = np.dot(f, c) / (np.linalg.norm(f) * np.linalg.norm(c) + 1e-10)
            return float(eu * (1.0 - cos_sim + 0.5))
        return eu

    def predict(self, features):
        """Predict for a single sample."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")

        return compute_openmax_scores(
            features, self.class_centroids, self.weibull_params,
            self.alpha_rank, self.distance_type
        )
