"""
Spider-Sense v2 - Multi-Agent Debate Layer
Three agents (Detector, Analyst, Arbiter) debate each anomaly detection.

Supports two modes:
- Local mode: Rule-based agents (no API needed)
- API mode: LLM-powered agents (GPT-4/Qwen API)
"""
import json
import os
import hashlib
import datetime
from typing import Dict, List, Optional


class BaseAgent:
    """Base class for debate agents."""

    def __init__(self, name: str, role: str, perspective: str):
        self.name = name
        self.role = role
        self.perspective = perspective

    def analyze(self, context: Dict) -> Dict:
        """Override in subclasses."""
        raise NotImplementedError


class RuleBasedAgent(BaseAgent):
    """
    Rule-based agent that uses heuristics and pattern matching.
    No API needed — works fully offline.
    """

    def __init__(self, name: str, role: str, perspective: str, rules: List[Dict]):
        super().__init__(name, role, perspective)
        self.rules = rules

    def analyze(self, context: Dict) -> Dict:
        detection = context.get('detection', {})
        flow_info = context.get('flow_info', {})
        rag_context = context.get('rag_context', [])
        xai_context = context.get('xai_context', {})

        # Evaluate each rule against the context
        triggered_rules = []
        evidence_points = []
        counter_evidence = []

        for rule in self.rules:
            score = self._evaluate_rule(rule, detection, flow_info, rag_context)
            if score > 0.5:
                triggered_rules.append({'rule': rule['name'], 'score': score})
                if rule.get('type') == 'evidence':
                    evidence_points.append(rule['description'])
                elif rule.get('type') == 'counter':
                    counter_evidence.append(rule['description'])

        # Generate verdict based on rule evaluation
        confidence = self._compute_confidence(triggered_rules)
        verdict = self._make_verdict(confidence, evidence_points, counter_evidence)

        return {
            'agent': self.name,
            'role': self.role,
            'verdict': verdict['conclusion'],
            'confidence': round(confidence, 3),
            'evidence': evidence_points[:3],
            'counter_evidence': counter_evidence[:3],
            'triggered_rules': [r['rule'] for r in triggered_rules],
            'analysis': verdict['analysis'],
        }

    def _evaluate_rule(self, rule: Dict, detection: Dict, flow_info: Dict,
                       rag_context: List[Dict]) -> float:
        """Evaluate a single rule against current context. Returns score 0-1."""
        conditions = rule.get('conditions', {})
        score = 0.0
        matched = 0
        total = 0

        for key, condition in conditions.items():
            total += 1
            if key == 'high_confidence' and detection.get('class_confidence', 0) > 0.8:
                matched += 1
            elif key == 'low_confidence' and detection.get('class_confidence', 0) < 0.5:
                matched += 1
            elif key == 'is_unknown' and detection.get('is_unknown'):
                matched += 1
            elif key == 'rag_similarity_high' and rag_context and rag_context[0].get('similarity', 0) > 0.7:
                matched += 1
            elif key == 'rag_similarity_low' and (not rag_context or rag_context[0].get('similarity', 0) < 0.3):
                matched += 1
            elif key == 'unknown_prob_high' and detection.get('unknown_prob', 0) > 0.6:
                matched += 1
            elif key == 'unknown_prob_low' and detection.get('unknown_prob', 0) < 0.3:
                matched += 1
            elif key == 'iso_anomaly' and detection.get('iso_anomaly'):
                matched += 1
            elif key == 'high_traffic' and flow_info.get('packet_length', 0) > 1000:
                matched += 1
            elif key == 'many_connections' and flow_info.get('connection_count', 0) > 50:
                matched += 1

        if total > 0:
            score = matched / total
        return score

    def _compute_confidence(self, triggered_rules: List[Dict]) -> float:
        if not triggered_rules:
            return 0.3
        return min(0.95, sum(r['score'] for r in triggered_rules) / len(triggered_rules))

    def _make_verdict(self, confidence: float, evidence: List[str],
                       counter_evidence: List[str]) -> Dict:
        if confidence > 0.8:
            conclusion = 'CONFIRMED_THREAT'
            analysis = '基于多维度特征分析，该告警具有高度可信性。'
        elif confidence > 0.5:
            conclusion = 'LIKELY_THREAT'
            analysis = '存在异常迹象但部分特征不够明确，建议进一步调查。'
        elif counter_evidence:
            conclusion = 'LIKELY_BENIGN'
            analysis = '发现显著的良性解释证据，可能是误报。'
        else:
            conclusion = 'UNCERTAIN'
            analysis = '当前证据不足，无法做出明确判断。'

        return {'conclusion': conclusion, 'analysis': analysis}


class LLMAgent(BaseAgent):
    """
    LLM-powered agent using external API (GPT-4/Qwen).
    """

    def __init__(self, name: str, role: str, perspective: str,
                 api_key: str = None, model_name: str = None):
        super().__init__(name, role, perspective)
        self.api_key = api_key
        self.model_name = model_name or 'gpt-4'

    def analyze(self, context: Dict) -> Dict:
        """Send analysis prompt to LLM API."""
        prompt = self._build_prompt(context)
        response = self._call_api(prompt)
        return self._parse_response(response)

    def _build_prompt(self, context: Dict) -> str:
        detection = context.get('detection', {})
        flow_info = context.get('flow_info', {})
        rag_context = context.get('rag_context', [])
        xai_context = context.get('xai_context', {})

        rag_text = ''
        if rag_context:
            rag_text = '\n'.join(
                f"- {r.get('name', 'Unknown')} (相似度: {r.get('similarity', 0):.2f}): {r.get('description', '')}"
                for r in rag_context[:3]
            )

        xai_text = ''
        if xai_context:
            top_feats = xai_context.get('top_features', [])
            if top_feats:
                xai_text = '关键特征: ' + ', '.join(
                    f"{f['name']}(重要性:{f['importance']:.4f})" for f in top_feats[:5]
                )

        return f"""你是一名网络安全分析专家，担任"{self.role}"角色。
视角: {self.perspective}

请分析以下网络入侵检测告警:

**检测结果:**
- 预测类别: {detection.get('prediction', 'N/A')}
- 置信度: {detection.get('class_confidence', 0):.2%}
- 是否为未知攻击: {detection.get('is_unknown', False)}
- 未知概率: {detection.get('unknown_prob', 0):.2%}

**流量信息:**
- 源IP: {flow_info.get('src_ip', 'N/A')}
- 目标IP: {flow_info.get('dst_ip', 'N/A')}
- 协议: {flow_info.get('protocol', 'N/A')}
- 包长度: {flow_info.get('packet_length', 0)}

**RAG知识库匹配:**
{rag_text or '无匹配的攻击模式'}

**XAI特征归因:**
{xai_text or '无特征归因数据'}

请从你的角色视角进行分析，输出JSON:
{{"verdict": "CONFIRMED_THREAT|LIKELY_THREAT|UNCERTAIN|LIKELY_BENIGN", "confidence": 0.0-1.0, "evidence": ["证据1",...], "counter_evidence": ["反证1",...], "analysis": "详细分析"}}"""

    def _call_api(self, prompt: str) -> str:
        """Call LLM API. Uses OpenAI-compatible endpoint."""
        try:
            import requests
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            data = {
                'model': self.model_name,
                'messages': [
                    {'role': 'system', 'content': '你是网络安全分析专家，始终输出JSON格式的分析结果。'},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.3,
                'max_tokens': 500,
            }
            # Use OpenAI endpoint or compatible (e.g. local Qwen)
            base_url = os.environ.get('LLM_API_BASE', 'https://api.openai.com/v1').rstrip('/')
            url = f'{base_url}/chat/completions'
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']

            body = resp.text[:1024]
            print(f"[LLM DEBUG] {resp.status_code} {url} model={self.model_name}")
            print(f"[LLM DEBUG] response body: {body}")
            return json.dumps({
                'verdict': 'UNCERTAIN',
                'confidence': 0.5,
                'evidence': [],
                'counter_evidence': [f'API call failed ({resp.status_code})'],
                'analysis': f'LLM API returned status {resp.status_code}: {body}',
            })
        except Exception as e:
            print(f"[LLM DEBUG] exception during API call: {str(e)}")
            return json.dumps({
                'verdict': 'UNCERTAIN',
                'confidence': 0.5,
                'evidence': [],
                'counter_evidence': [str(e)],
                'analysis': 'LLM API call failed',
            })

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response JSON."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                return {
                    'agent': self.name,
                    'role': self.role,
                    'verdict': parsed.get('verdict', 'UNCERTAIN'),
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'evidence': parsed.get('evidence', []),
                    'counter_evidence': parsed.get('counter_evidence', []),
                    'analysis': parsed.get('analysis', ''),
                    'llm_model': self.model_name,
                }
        except (json.JSONDecodeError, KeyError) as e:
            pass

        return {
            'agent': self.name,
            'role': self.role,
            'verdict': 'UNCERTAIN',
            'confidence': 0.5,
            'evidence': [],
            'counter_evidence': [],
            'analysis': f'Failed to parse LLM response',
            'raw_response': response[:200],
        }


class AgentLayer:
    """
    Multi-agent debate layer.

    Three agents debate each anomaly:
    - Detector: Focuses on attack signatures and technical evidence
    - Analyst: Looks for benign explanations and false positive signals
    - Arbiter: Weighs both sides and makes final determination
    """

    def __init__(self, rag_engine=None, use_api=False, api_key=None, model_name=None):
        self.rag_engine = rag_engine
        self.use_api = use_api
        self.api_key = api_key
        self.model_name = model_name

        # Initialize agents
        if use_api and api_key:
            self._init_llm_agents()
        else:
            self._init_rule_agents()

    def _init_rule_agents(self):
        """Initialize rule-based agents."""
        detector_rules = [
            {'name': 'high_conf_match', 'type': 'evidence',
             'conditions': {'high_confidence': True},
             'description': '检测模型输出高置信度分类'},
            {'name': 'rag_pattern_match', 'type': 'evidence',
             'conditions': {'rag_similarity_high': True},
             'description': 'RAG知识库匹配到高度相似的攻击模式'},
            {'name': 'unknown_detection', 'type': 'evidence',
             'conditions': {'is_unknown': True},
             'description': '模型判定为未知攻击类型（开放集识别触发）'},
            {'name': 'iso_anomaly_detected', 'type': 'evidence',
             'conditions': {'iso_anomaly': True},
             'description': 'Isolation Forest统计异常检测器触发'},
            {'name': 'high_unknown_prob', 'type': 'evidence',
             'conditions': {'unknown_prob_high': True},
             'description': '开放集未知概率较高'},
        ]

        analyst_rules = [
            {'name': 'low_confidence', 'type': 'counter',
             'conditions': {'low_confidence': True},
             'description': '模型置信度较低，可能是误报'},
            {'name': 'no_rag_match', 'type': 'counter',
             'conditions': {'rag_similarity_low': True},
             'description': '未匹配到已知攻击模式，可能是良性异常'},
            {'name': 'low_unknown_prob', 'type': 'counter',
             'conditions': {'unknown_prob_low': True},
             'description': '未知概率较低，可能仍在正常范围内'},
            {'name': 'benign_traffic', 'type': 'counter',
             'conditions': {'high_traffic': True},
             'description': '流量特征可能由正常大流量业务引起'},
        ]

        arbiter_rules = [
            {'name': 'multi_evidence', 'type': 'weigh',
             'conditions': {'high_confidence': True, 'rag_similarity_high': True},
             'description': '多维度证据一致支持威胁判定'},
            {'name': 'conflicting_signals', 'type': 'weigh',
             'conditions': {'low_confidence': True, 'unknown_prob_high': True},
             'description': '检测信号存在矛盾，需谨慎判断'},
        ]

        self.detector = RuleBasedAgent('detector', '🕵️ 检测员',
                                        '从攻击特征和技术证据角度分析，识别威胁信号',
                                        detector_rules)
        self.analyst = RuleBasedAgent('analyst', '🔬 分析师',
                                       '从误报角度审视，寻找良性解释和反证',
                                       analyst_rules)
        self.arbiter = RuleBasedAgent('arbiter', '⚖️ 裁决官',
                                       '综合双方意见，做出最终判定并给出处置建议',
                                       arbiter_rules)

    def _init_llm_agents(self):
        """Initialize LLM-powered agents."""
        self.detector = LLMAgent('detector', '🕵️ 检测员',
                                  '从攻击特征和技术证据角度分析，识别威胁信号',
                                  self.api_key, self.model_name)
        self.analyst = LLMAgent('analyst', '🔬 分析师',
                                 '从误报角度审视，寻找良性解释和反证',
                                 self.api_key, self.model_name)
        self.arbiter = LLMAgent('arbiter', '⚖️ 裁决官',
                                 '综合双方意见，做出最终判定并给出处置建议',
                                 self.api_key, self.model_name)

    def debate(self, detection_result: Dict, flow_info: Dict,
               rag_context: Optional[List[Dict]] = None,
               xai_context: Optional[Dict] = None) -> Dict:
        """
        Run the 3-agent debate on a detection.

        Args:
            detection_result: Output from DetectionEngine.predict()
            flow_info: Network flow metadata
            rag_context: RAG search results (from RAGEngine.search())
            xai_context: XAI explanation (from XAIEngine.explain())

        Returns:
            Dict with agent opinions, voting result, and final verdict
        """
        context = {
            'detection': detection_result,
            'flow_info': flow_info,
            'rag_context': rag_context or [],
            'xai_context': xai_context or {},
        }

        # All three agents analyze independently
        detector_opinion = self.detector.analyze(context)
        analyst_opinion = self.analyst.analyze(context)
        arbiter_opinion = self.arbiter.analyze(context)

        # Voting logic
        verdicts = [
            detector_opinion.get('verdict', 'UNCERTAIN'),
            analyst_opinion.get('verdict', 'UNCERTAIN'),
            arbiter_opinion.get('verdict', 'UNCERTAIN'),
        ]

        # Count threat votes
        threat_votes = sum(1 for v in verdicts if v in ('CONFIRMED_THREAT', 'LIKELY_THREAT'))

        if threat_votes >= 2:
            final_verdict = 'MALICIOUS'
            action = '建议立即阻断源IP，启动应急响应流程'
        elif threat_votes == 1:
            final_verdict = 'SUSPICIOUS'
            action = '建议持续监控该流量源，收集更多证据后研判'
        else:
            final_verdict = 'BENIGN'
            action = '疑似误报，建议标记为白名单候选'

        # Generate recommendation
        recommendation = self._generate_recommendation(
            final_verdict, detection_result, flow_info, rag_context
        )

        return {
            'verdict': final_verdict,
            'vote_count': f'{threat_votes}/3 判定为威胁',
            'action': action,
            'recommendation': recommendation,
            'detector': detector_opinion,
            'analyst': analyst_opinion,
            'arbiter': arbiter_opinion,
            'rag_context_used': bool(rag_context and len(rag_context) > 0),
            'xai_context_used': bool(xai_context),
            'debate_mode': 'llm' if self.use_api else 'rule_based',
            'timestamp': datetime.datetime.now().isoformat(),
        }

    def _generate_recommendation(self, verdict: str, detection: Dict,
                                  flow_info: Dict, rag_context: List[Dict]) -> str:
        """Generate actionable recommendation based on debate outcome."""
        if verdict == 'MALICIOUS':
            src_ip = flow_info.get('src_ip', '未知来源')
            rec = f'🚨 确认为恶意流量。建议: 1) 立即封锁源IP {src_ip}；'
            rec += '2) 检查目标主机是否已被入侵；3) 收集完整pcap证据；4) 上报安全运营中心。'
            if rag_context:
                mitre_ids = [r.get('mitre', {}).get('id', '') for r in rag_context if r.get('mitre')]
                if mitre_ids:
                    rec += f' 相关MITRE ATT&CK技术: {", ".join(mitre_ids)}。'
            return rec
        elif verdict == 'SUSPICIOUS':
            return ('⚠️ 可疑流量需进一步研判。建议: 1) 对该源IP启用增强监控；'
                    '2) 记录并关联其他安全日志；3) 如果持续可疑则升级为恶意。')
        else:
            return ('✅ 判定为良性/误报。建议: 1) 将特征加入白名单候选；'
                    '2) 如果是高频误报，考虑调整检测灵敏度。')
