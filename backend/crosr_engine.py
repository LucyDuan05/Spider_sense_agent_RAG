"""
CROSR Detection Engine: DHRNet-1D feature extraction + WeibullOpenMax.
Uses the ORIGINAL CROSR model (reconstruction loss) with our pure-Python OpenMax.
"""
import numpy as np
import pickle
import os
import sys
import torch
import torch.nn as nn

# Handle PyInstaller paths
if getattr(sys, 'frozen', False):
    if sys._MEIPASS not in sys.path:
        sys.path.insert(0, sys._MEIPASS)

from DHR_Net_1D import DHRNet1D
from backend.weibull_openmax import WeibullOpenMax


class CROSREngine:
    """Unified detection engine for Spider-Sense v2."""

    def __init__(self):
        self.model = None
        self.om_detector = None
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.num_classes = 6
        self.input_dim = 78
        self.label_names = ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'SSH-Patator']
        self.loaded = False

    def load(self, model_path='models/weibull_om/model.pth',
             detector_path='models/weibull_om/detector.pkl'):
        """Load DHRNet model and WeibullOpenMax detector."""
        # Load model
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)

        self.num_classes = ckpt.get('num_classes', 6)
        self.label_names = list(ckpt.get('label_names',
            ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'SSH-Patator']))

        self.model = DHRNet1D(
            num_classes=self.num_classes,
            input_channels=ckpt.get('input_channels', 1),
            base_channels=ckpt.get('base_channels', 128),
            hidden_dim=ckpt.get('hidden_dim', 512),
        )
        self.model.load_state_dict(sd)
        self.model.eval()

        # Load OpenMax detector
        if os.path.exists(detector_path):
            with open(detector_path, 'rb') as f:
                self.om_detector = pickle.load(f)
            print(f"[✓] WeibullOpenMax loaded: {len(self.om_detector.class_centroids)} classes")

        self.loaded = True
        print(f"[✓] CROSR Engine ready: {self.num_classes} classes, {self.label_names}")
        return self

    def extract_features(self, raw_features):
        """Extract CROSR features. Returns dict with 'embedding' key for orchestrator compat."""
        if isinstance(raw_features, np.ndarray):
            raw_features = torch.from_numpy(raw_features).float()
        if raw_features.dim() == 1:
            raw_features = raw_features.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
        elif raw_features.dim() == 2:
            raw_features = raw_features.unsqueeze(1)  # [batch, 1, features]

        with torch.no_grad():
            logits, _, latent = self.model(raw_features)
        pooled = [self.pool(z).flatten(start_dim=1) for z in latent]
        features = torch.cat([logits] + pooled, dim=1)

        # Return dict format (compatible with orchestrator)
        probs = torch.softmax(logits, dim=-1)
        return {
            'embedding': features.numpy(),
            'logits': logits.numpy(),
            'probabilities': probs.numpy(),
        }

    def predict(self, raw_features, return_details=True):
        """Full detection pipeline."""
        raw = np.asarray(raw_features, dtype=np.float32).flatten()

        # Extract CROSR features
        out = self.extract_features(raw)
        feat_vec = out['embedding'][0]  # single sample feature
        logit_vec = out['logits'][0]
        probs = out['probabilities'][0]  # softmax probabilities

        # Classification
        pred_class = int(np.argmax(logit_vec))
        class_conf = float(probs[pred_class])  # probability, not raw logit

        # OpenMax scoring
        if self.om_detector and self.om_detector.fitted:
            om_result = self.om_detector.predict(feat_vec)
            is_unknown = om_result['is_unknown']
            unknown_score = om_result['unknown_score']
            if is_unknown:
                prediction = 'UNKNOWN'
            else:
                cls = om_result['predicted_class']
                prediction = str(self.label_names[cls]) if cls < len(self.label_names) else f'class_{cls}'
        else:
            is_unknown = False
            unknown_score = 0.0
            prediction = str(self.label_names[pred_class]) if pred_class < len(self.label_names) else f'class_{pred_class}'
            om_result = {}

        result = {
            'prediction': prediction,
            'predicted_class': pred_class,
            'class_confidence': round(float(class_conf), 4),
            'is_unknown': bool(is_unknown),
            'is_anomaly': bool(is_unknown),
            'iso_anomaly': False,
        }

        if return_details:
            result['unknown_prob'] = round(float(unknown_score), 4)
            result['unknown_score'] = round(float(unknown_score), 4)
            result['openmax_scores'] = {str(k): round(float(v), 4)
                                        for k, v in om_result.get('openmax_scores', {}).items()}
            result['weibull_probs'] = {str(k): round(float(v), 4)
                                       for k, v in om_result.get('weibull_probs', {}).items()}
            result['probabilities'] = {str(self.label_names[i]) if i < len(self.label_names) else f'class_{i}':
                                       round(float(p), 6) for i, p in enumerate(probs)}
            result['all_distances'] = {str(k): round(float(v), 4)
                                       for k, v in om_result.get('all_distances', {}).items()}

        return result

    def batch_predict(self, features_batch):
        """Batch prediction."""
        return [self.predict(f) for f in features_batch]

    def get_stats(self):
        return {
            'model_loaded': self.loaded,
            'num_classes': self.num_classes,
            'label_names': self.label_names,
            'om_fitted': self.om_detector is not None and self.om_detector.fitted,
        }
