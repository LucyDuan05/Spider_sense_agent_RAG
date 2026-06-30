"""
Spider-Sense v2 - XAI Engine
Explainable AI: feature attribution + natural language explanation templates.
"""
import numpy as np
from typing import Dict, List, Optional


class XAIEngine:
    """
    Lightweight XAI engine for explaining model predictions.

    Uses:
    - Gradient-based attribution (for Transformer model)
    - Feature importance ranking
    - Template-based natural language generation
    """

    def __init__(self):
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate',
            # Extended features
            'flow_duration', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean',
            'flow_byts_s', 'flow_pkts_s', 'fwd_iat_mean', 'bwd_iat_mean',
            'pkt_len_var', 'pkt_len_std', 'syn_flag_cnt', 'ack_flag_cnt',
            'fin_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'urg_flag_cnt',
            'down_up_ratio', 'init_win_bytes_fwd', 'init_win_bytes_bwd',
            'active_mean', 'idle_mean', 'active_max', 'idle_max',
            'subflow_fwd_pkts', 'subflow_bwd_pkts', 'subflow_fwd_byts',
        ]

    def explain(self, features: np.ndarray, engine) -> Dict:
        """
        Generate XAI explanation for a prediction.

        Args:
            features: Input feature vector
            engine: DetectionEngine instance

        Returns:
            Dict with feature importance, top contributing features, and NL explanation
        """
        features = np.asarray(features, dtype=np.float32).flatten()
        result = engine.predict(features, return_details=True)

        # Feature importance via simple gradient approach
        feature_importance = self._compute_feature_importance(features, engine)

        # Get top contributing features
        top_features = self._get_top_features(feature_importance, top_k=8)

        # Generate natural language explanation
        explanation = self._generate_explanation(
            prediction=result['prediction'],
            is_unknown=result['is_unknown'],
            confidence=result['class_confidence'],
            top_features=top_features,
            unknown_prob=result.get('unknown_prob', 0),
        )

        return {
            'prediction': result['prediction'],
            'confidence': result['class_confidence'],
            'is_unknown': result['is_unknown'],
            'feature_importance': feature_importance[:20],  # Top 20 features
            'top_features': top_features,
            'explanation': explanation,
            'method': 'gradient_attribution',
        }

    def _compute_feature_importance(self, features: np.ndarray, engine) -> List[Dict]:
        """
        Compute feature importance via perturbation-based approach.
        Simpler and more robust than gradient methods for this use case.
        """
        features = features.copy()
        baseline_result = engine.predict(features, return_details=True)
        baseline_conf = baseline_result['class_confidence']

        importance_scores = []
        for i in range(min(len(features), len(self.feature_names))):
            # Perturb this feature (zero it out and see impact)
            perturbed = features.copy()
            perturbed[i] = 0.0

            perturbed_result = engine.predict(perturbed, return_details=True)
            perturbed_conf = perturbed_result['class_confidence']

            # Importance = drop in confidence
            importance = abs(baseline_conf - perturbed_conf)

            name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            importance_scores.append({
                'index': i,
                'name': name,
                'value': round(float(features[i]), 6),
                'importance': round(float(importance), 6),
            })

        # Sort by importance descending
        importance_scores.sort(key=lambda x: x['importance'], reverse=True)
        return importance_scores

    def _get_top_features(self, importance_scores: List[Dict], top_k: int = 8) -> List[Dict]:
        """Get top-K most important features."""
        return [f for f in importance_scores[:top_k] if f['importance'] > 0.001]

    def _generate_explanation(self, prediction: str, is_unknown: bool,
                               confidence: float, top_features: List[Dict],
                               unknown_prob: float) -> str:
        """Generate natural language explanation from feature attributions."""
        if not top_features:
            return f"模型预测为 {prediction}，置信度 {confidence:.1%}。无显著特征归因可用。"

        # Build feature description
        feature_desc = "、".join(
            f"{f['name']}(贡献度{f['importance']:.4f})"
            for f in top_features[:5]
        )

        if is_unknown:
            return (
                f"⚠️ 检测到未知攻击类型 (开放集判定)。模型置信度 {confidence:.1%}，"
                f"未知概率 {unknown_prob:.1%}。"
                f"主要异常特征: {feature_desc}。"
                f"该流量模式与已知攻击类型不匹配，建议进行深度分析。"
            )
        elif confidence < 0.7:
            return (
                f"⚠️ 预测为 {prediction}，但置信度较低 ({confidence:.1%})。"
                f"主要特征: {feature_desc}。"
                f"建议结合多智能体研判确认。"
            )
        else:
            return (
                f"✅ 预测为 {prediction}，置信度 {confidence:.1%}。"
                f"主要支持特征: {feature_desc}。"
            )

    def explain_batch(self, features_batch: np.ndarray, engine) -> List[Dict]:
        """Generate explanations for a batch of samples."""
        return [self.explain(f, engine) for f in features_batch]
