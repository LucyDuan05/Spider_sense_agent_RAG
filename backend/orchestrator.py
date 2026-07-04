"""
Spider-Sense v2 - Orchestrator
Coordinates detection → Flow-to-Text → Semantic RAG → Agent → XAI pipeline
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
from .flow_to_text import FlowToText
from .semantic_rag import SemanticRAG


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
                 xai: Optional[XAIEngine] = None,
                 semantic_rag: Optional[SemanticRAG] = None,
                 flow_to_text: Optional[FlowToText] = None,
                 scaler_path: str = None):
        self.engine = engine
        self.rag = rag or RAGEngine()        # MITRE 知识 + 历史追踪
        self.xai = xai or XAIEngine()

        # Semantic RAG: flow → text → embedding → MITRE search
        self.semantic_rag = semantic_rag
        self.flow_to_text = flow_to_text

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

        # Step 2: Semantic RAG — flow→text→embedding→MITRE search
        rag_result = None
        if need_analysis and self.semantic_rag:
            try:
                query_text = self.flow_to_text.to_rag_query(features, detection)
                rag_result = self.semantic_rag.search(query_text, top_k=3)
            except Exception:
                rag_result = None

        # Step 3: XAI explanation (for all anomalies)
        xai_result = None
        if need_analysis:
            xai_result = self.xai.explain(features, self.engine)

        # Step 4: RAG hint — UNKNOWN 时附加最可能的类别（过滤正常流量锚点）
        rag_hint = None
        if detection.get('is_unknown') and rag_result and len(rag_result) > 0:
            top = rag_result[0]
            top_id = top.get('mitre', {}).get('id', '')
            # 跳过正常流量锚点 — 不显示攻击标签
            if top_id and top_id != 'BENIGN':
                rag_hint = f"{top_id} {top.get('name', '')}"
            elif top.get('similarity', 0) < 0.4:
                rag_hint = None  # 相似度过低, 不确定
            else:
                rag_hint = None  # 匹配到正常流量, 不显示标签

        # Step 5: Assemble result (NO agent — 按需触发)
        elapsed = time.time() - start_time

        result = {
            'timestamp': datetime.now().isoformat(),
            'flow_info': flow_info or {},
            'detection': detection,
            'rag': rag_result,
            'rag_hint': rag_hint,
            'xai': xai_result,
            'agent': None,  # 不再自动运行 Agent, 由 analyze() 按需填充
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

    def analyze(self, features: np.ndarray, flow_info: Optional[Dict] = None,
                detection_result: Optional[Dict] = None) -> Dict:
        """
        按需运行完整 Agent 分析（RAG + XAI + Agent 辩论）。
        仅在用户点击某条事件时调用，不自动运行。
        """
        if not self.agent_layer:
            return {'error': 'Agent层未初始化', 'verdict': 'AGENT_UNAVAILABLE'}

        # 如果没有传入 detection_result, 先跑检测
        if detection_result is None:
            detection_result = self.engine.predict(features, return_details=True)

        # Semantic RAG — flow→text→embedding→MITRE search
        rag_result = None
        if self.semantic_rag:
            try:
                query_text = self.flow_to_text.to_rag_query(features, detection_result)
                rag_result = self.semantic_rag.search(query_text, top_k=3)
            except Exception:
                rag_result = None

        # XAI (如果还没跑过)
        xai_result = None
        if detection_result.get('anomaly_type', 'unknown') in ('known_attack', 'unknown'):
            xai_result = self.xai.explain(features, self.engine)

        # Agent 辩论 (这是唯一消耗 token 的步骤)
        agent_result = None
        try:
            agent_result = self.agent_layer.debate(
                detection_result=detection_result,
                flow_info=flow_info or {},
                rag_context=rag_result,
                xai_context=xai_result,
                rounds=getattr(self.agent_layer, 'debate_rounds', 1),
            )
            self.agent_cache.append(agent_result)
        except Exception as e:
            agent_result = {'error': str(e), 'verdict': 'AGENT_ERROR'}

        return {
            'rag': rag_result,
            'xai': xai_result,
            'agent': agent_result,
        }

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
