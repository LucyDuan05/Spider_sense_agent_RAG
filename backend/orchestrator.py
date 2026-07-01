"""
Spider-Sense v2 - Orchestrator
Coordinates detection → Agent analysis → RAG → XAI pipeline
"""
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any

from .detection_engine import DetectionEngine
from .rag_engine import RAGEngine
from .xai_engine import XAIEngine


class Orchestrator:
    """
    Central orchestrator that coordinates the full detection pipeline:

    1. Traffic → Feature Extraction → Detection Engine
    2. Anomaly detected → Multi-Agent Debate (if enabled)
    3. Enrich with RAG knowledge base
    4. Generate XAI explanation
    5. Assemble final report
    """

    def __init__(self, engine: DetectionEngine,
                 rag: Optional[RAGEngine] = None,
                 xai: Optional[XAIEngine] = None):
        self.engine = engine
        self.rag = rag or RAGEngine()
        self.xai = xai or XAIEngine()

        # Agent layer (lazy import to avoid circular deps)
        self.agent_layer = None

        # State
        self.is_running = False
        self.lock = threading.Lock()
        self.total_count = 0
        self.class_counts = {}
        self.unknown_probs = []
        self.recent_results = deque(maxlen=200)
        self.latest_prediction = 'N/A'
        self.latest_flow_info = None
        self.session_start = None

        # Agent analysis cache
        self.agent_cache = deque(maxlen=50)  # Cache recent agent analyses

    def init_agent_layer(self, use_api=False, api_key=None, model_name=None):
        """Initialize the multi-agent debate layer."""
        from .agent_layer import AgentLayer
        self.agent_layer = AgentLayer(
            rag_engine=self.rag,
            use_api=use_api,
            api_key=api_key,
            model_name=model_name
        )

    def process_flow(self, features: np.ndarray, flow_info: Optional[Dict] = None) -> Dict:
        """
        Process a single network flow through the full pipeline.

        Args:
            features: Feature vector [input_dim]
            flow_info: Optional dict with src_ip, dst_ip, protocol, etc.

        Returns:
            Complete analysis result dict
        """
        start_time = time.time()

        # Step 1: Detection
        detection = self.engine.predict(features, return_details=True)

        # ── 分层分析策略 ────────────────────────────────────────────
        #
        # anomaly_type  'benign'        → 真正良性, 跳过分析
        #               'known_attack'  → 已知攻击, 全量分析
        #               'unknown'       → 零日/未知攻击, 全量分析
        #
        need_analysis = detection.get('anomaly_type', 'unknown') in ('known_attack', 'unknown')

        # Step 2: RAG enrichment (用 predict 已算好的缓存 embedding, 避免二次前向)
        rag_result = None
        if need_analysis:
            cached_emb = detection.get('embedding')
            if cached_emb:
                rag_result = self.rag.search(
                    embedding=np.array(cached_emb, dtype=np.float32),
                    top_k=3
                )

        # Step 3: XAI explanation (for all anomalies)
        xai_result = None
        if need_analysis:
            xai_result = self.xai.explain(features, self.engine)

        # Step 4: Agent debate (for all anomalies)
        agent_result = None
        if need_analysis and self.agent_layer:
            try:
                agent_result = self.agent_layer.debate(
                    detection_result=detection,
                    flow_info=flow_info or {},
                    rag_context=rag_result,
                    xai_context=xai_result,
                )
                self.agent_cache.append(agent_result)
            except Exception as e:
                agent_result = {'error': str(e), 'verdict': 'AGENT_ERROR'}

        # Step 5: Assemble result
        elapsed = time.time() - start_time

        result = {
            'timestamp': datetime.now().isoformat(),
            'flow_info': flow_info or {},
            'detection': detection,
            'rag': rag_result,
            'xai': xai_result,
            'agent': agent_result,
            'pipeline_time_ms': round(elapsed * 1000, 2),
        }

        # Update state
        with self.lock:
            self.total_count += 1
            prediction = detection['prediction']
            self.class_counts[prediction] = self.class_counts.get(prediction, 0) + 1
            if detection.get('unknown_prob'):
                self.unknown_probs.append(detection['unknown_prob'])
                if len(self.unknown_probs) > 1000:
                    self.unknown_probs = self.unknown_probs[-1000:]
            self.recent_results.appendleft(result)
            self.latest_prediction = prediction
            self.latest_flow_info = flow_info

        return result

    def batch_process(self, features_list: List[np.ndarray],
                      flow_info_list: Optional[List[Dict]] = None) -> List[Dict]:
        """Batch process multiple flows."""
        results = []
        flow_info_list = flow_info_list or [None] * len(features_list)
        for features, flow_info in zip(features_list, flow_info_list):
            results.append(self.process_flow(features, flow_info))
        return results

    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        with self.lock:
            unknown_rate = 0.0
            if self.unknown_probs:
                unknown_rate = sum(1 for p in self.unknown_probs if p >= 0.5) / len(self.unknown_probs)

            return {
                'isRunning': self.is_running,
                'prediction': self.latest_prediction,
                'flowDetail': self.latest_flow_info,
                'recent': list(self.recent_results)[:20],
                'totalCount': self.total_count,
                'classCounts': dict(self.class_counts),
                'stats': {
                    'unknown_rate': round(unknown_rate, 4),
                    'unknown_prob_mean': round(float(np.mean(self.unknown_probs)), 4) if self.unknown_probs else None,
                    'unknown_prob_max': round(float(np.max(self.unknown_probs)), 4) if self.unknown_probs else None,
                    'unknown_prob_p95': round(float(np.percentile(self.unknown_probs, 95)), 4) if len(self.unknown_probs) >= 20 else None,
                }
            }

    def get_session_stats(self) -> Dict:
        """Get session-level statistics."""
        with self.lock:
            return {
                'openmax_enabled': self.engine.open_set_detector is not None and self.engine.open_set_detector.fitted,
                'unknown_rate': self.unknown_probs and sum(1 for p in self.unknown_probs if p >= 0.5) / len(self.unknown_probs) or 0,
                'unknown_prob_mean': float(np.mean(self.unknown_probs)) if self.unknown_probs else None,
                'unknown_prob_max': float(np.max(self.unknown_probs)) if self.unknown_probs else None,
                'unknown_prob_p95': float(np.percentile(self.unknown_probs, 95)) if len(self.unknown_probs) >= 20 else None,
                'total_analyzed': self.total_count,
                'class_counts': dict(self.class_counts),
                'recent_agent_analyses': list(self.agent_cache)[-10:],
            }

    def reset_stats(self):
        """Reset session statistics."""
        with self.lock:
            self.class_counts = {}
            self.unknown_probs = []
            self.total_count = 0
            self.agent_cache.clear()
