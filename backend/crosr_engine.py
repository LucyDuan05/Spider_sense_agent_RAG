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
            logits, recon, latent = self.model(raw_features)
        pooled = [self.pool(z).flatten(start_dim=1) for z in latent]
        features = torch.cat([logits] + pooled, dim=1)

        # Compute reconstruction error (MSE between input and decoder output)
        raw_flat = raw_features.view(raw_features.size(0), -1)
        recon_flat = recon.view(recon.size(0), -1)
        # Trim to same length if needed
        min_dim = min(raw_flat.size(-1), recon_flat.size(-1))
        recon_error = torch.nn.functional.mse_loss(
            raw_flat[:, :min_dim], recon_flat[:, :min_dim], reduction='none'
        ).mean(dim=1)  # [batch]

        # Return dict format (compatible with orchestrator)
        probs = torch.softmax(logits, dim=-1)
        return {
            'embedding': features.numpy(),
            'logits': logits.numpy(),
            'probabilities': probs.numpy(),
            'reconstruction': recon.numpy(),
            'recon_error': recon_error.numpy(),
        }

    def predict(self, raw_features, return_details=True):
        """Full detection pipeline."""
        raw = np.asarray(raw_features, dtype=np.float32).flatten()

        # Extract CROSR features (includes reconstruction error now)
        out = self.extract_features(raw)
        feat_vec = out['embedding'][0]  # single sample feature
        logit_vec = out['logits'][0]
        probs = out['probabilities'][0]  # softmax probabilities
        recon_error = float(out['recon_error'][0])

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

        # ── 融合异常判定: 四象限模型 ──────────────────────────────
        #
        #   is_unknown: OpenMax 无法将样本归入任何已知 6 类
        #   is_anomaly: 综合多信号判定"需要深入分析"
        #
        #   三个正交异常信号:
        #   ① unknown_score    → 距已知类分布过远 (OpenMax)
        #   ② class_confidence → 模型不确定自己的判断 (Softmax)
        #   ③ recon_error      → 解码器无法重建输入 (CROSR)
        #
        #   四象限:
        #                  Normal (正常)         Anomalous (异常)
        #   Known (已知)  ① BENIGN, 置信度≥0.7   ② 已知攻击 (DDoS 95%)
        #                   → 跳过分析             → 全量分析 (本次修复)
        #   Unknown (未知) ③ 新型良性(低概率)      ④ 零日攻击
        #                   → 保守标记 unknown      → 全量分析
        #
        #   关键: benign 必须同时检查 prediction == 'BENIGN' 和高置信度。
        #   DDoS 95% 满足高置信度但不满足预测类别, 不会误判为 benign。
        #
        confident_enough = class_conf >= 0.7
        recon_abnormal = recon_error > 0.15     # 阈值可调, 需根据训练集统计
        openmax_alarmed = unknown_score > 0.5

        is_genuinely_benign = (
            (prediction == 'BENIGN')            # ① DHRNet 明确预测为 BENIGN
            and confident_enough                # ② 高置信度
            and not openmax_alarmed             # ③ OpenMax 未报警
            and not recon_abnormal              # ④ 重建误差低
        )

        is_anomaly = not is_genuinely_benign

        if is_genuinely_benign:
            anomaly_type = 'benign'             # ① 真正良性 — 跳过分析
        elif is_unknown:
            anomaly_type = 'unknown'            # ④ 零日/未知攻击 — 全量分析
        else:
            anomaly_type = 'known_attack'       # ②③ 已知攻击/异常 — 全量分析

        result = {
            'prediction': prediction,
            'predicted_class': pred_class,
            'class_confidence': round(float(class_conf), 4),
            'is_unknown': bool(is_unknown),
            'is_anomaly': bool(is_anomaly),
            'anomaly_type': anomaly_type,
            'anomaly_score': round(float(
                0.4 * unknown_score +
                0.4 * (1.0 - class_conf) +
                0.2 * min(1.0, recon_error * 5)
            ), 4),
            'recon_error': round(float(recon_error), 6),
            'embedding': feat_vec.tolist(),   # 缓存, 供 orchestrator 直接用
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
