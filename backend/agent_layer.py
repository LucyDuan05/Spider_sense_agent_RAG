"""
Spider-Sense v2 - Multi-Agent Debate Layer
Three agents (Detector, Analyst, Arbiter) debate each anomaly detection.

Supports:
- Local mode: Rule-based agents (no API needed)
- API mode: LLM-powered agents (GPT-4/Ollama/Qwen)
- Multi-round debate with weighted verdict
"""
import json
import os
import hashlib
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# ── 规则的组合条件操作符 ───────────────────────────────────────────

def _eval_condition(metric_value: Any, op: str, target: Any) -> bool:
    """Evaluate a single condition: metric_value op target."""
    if op == '>': return float(metric_value) > float(target)
    elif op == '>=': return float(metric_value) >= float(target)
    elif op == '<': return float(metric_value) < float(target)
    elif op == '<=': return float(metric_value) <= float(target)
    elif op == '==': return metric_value == target
    elif op == '!=': return metric_value != target
    elif op == 'in': return metric_value in target
    elif op == 'nin':  # not in
        return metric_value not in target
    return False


def _get_metric(context: Dict, metric: str) -> Any:
    """Extract a metric value from the context dict using dot notation."""
    detection = context.get('detection', {})
    flow_info = context.get('flow_info', {})
    rag_context = context.get('rag_context', [])
    xai_context = context.get('xai_context', {})

    metrics = {
        'class_confidence': detection.get('class_confidence', 0),
        'prediction': detection.get('prediction', ''),
        'is_unknown': bool(detection.get('is_unknown', False)),
        'is_anomaly': bool(detection.get('is_anomaly', False)),
        'anomaly_type': detection.get('anomaly_type', ''),
        'unknown_score': detection.get('unknown_prob', detection.get('unknown_score', 0)),
        'recon_error': detection.get('recon_error', 0),
        'anomaly_score': detection.get('anomaly_score', 0),
        'packet_length': flow_info.get('packet_length', 0),
        'protocol': flow_info.get('protocol', ''),
        'connection_count': flow_info.get('connection_count', 0),
        'rag_top_similarity': rag_context[0].get('similarity', 0) if rag_context else 0,
        'rag_top_name': rag_context[0].get('name', '') if rag_context else '',
        'rag_has_match': bool(rag_context and len(rag_context) > 0),
        'xai_available': bool(xai_context and xai_context.get('top_features')),
    }
    return metrics.get(metric, 0)


# ── 基础 Agent ───────────────────────────────────────────────────────

class BaseAgent:
    """Base class for debate agents."""

    def __init__(self, name: str, role: str, perspective: str):
        self.name = name
        self.role = role
        self.perspective = perspective

    def analyze(self, context: Dict) -> Dict:
        raise NotImplementedError


class RuleBasedAgent(BaseAgent):
    """
    Rule-based agent using weighted combination conditions.
    No API needed — works fully offline.
    """

    def __init__(self, name: str, role: str, perspective: str, rules: List[Dict]):
        super().__init__(name, role, perspective)
        self.rules = rules

    def analyze(self, context: Dict) -> Dict:
        """Evaluate all rules against context. Returns opinion dict."""
        total_weight = 0.0
        matched_weight = 0.0
        evidence_points = []
        counter_evidence = []
        triggered = []

        for rule in self.rules:
            score = self._evaluate_rule(rule, context)
            weight = abs(rule.get('weight', 0.5))
            total_weight += weight

            if score > 0:
                matched_weight += weight * score
                triggered.append(rule['name'])
                if rule.get('type') == 'evidence':
                    evidence_points.append(rule['description'])
                elif rule.get('type') == 'counter':
                    counter_evidence.append(rule['description'])

        confidence = matched_weight / max(total_weight, 0.001)
        confidence = min(max(confidence, 0.0), 1.0)
        verdict = self._make_verdict(confidence, evidence_points, counter_evidence)

        return {
            'agent': self.name,
            'role': self.role,
            'verdict': verdict['conclusion'],
            'confidence': round(confidence, 3),
            'evidence': evidence_points[:5],
            'counter_evidence': counter_evidence[:5],
            'triggered_rules': triggered,
            'analysis': verdict['analysis'],
        }

    def _evaluate_rule(self, rule: Dict, context: Dict) -> float:
        """Evaluate a weighted rule. Returns match score 0.0-1.0."""
        conditions = rule.get('conditions', [])
        if not conditions:
            return 1.0  # No conditions = always match

        matched = 0
        for cond in conditions:
            metric = cond.get('metric', '')
            op = cond.get('op', '==')
            target = cond.get('value', True)
            val = _get_metric(context, metric)
            if _eval_condition(val, op, target):
                matched += 1

        return matched / len(conditions)

    def _make_verdict(self, confidence: float, evidence: List[str],
                       counter_evidence: List[str]) -> Dict:
        if confidence > 0.7:
            return {'conclusion': 'CONFIRMED_THREAT',
                    'analysis': '基于多维度特征分析，该告警具有高度可信性。'}
        elif confidence > 0.4:
            return {'conclusion': 'LIKELY_THREAT',
                    'analysis': '存在异常迹象但部分特征不够明确，建议进一步调查。'}
        elif counter_evidence and len(counter_evidence) >= len(evidence):
            return {'conclusion': 'LIKELY_BENIGN',
                    'analysis': '发现显著的良性解释证据，可能是误报。'}
        else:
            return {'conclusion': 'UNCERTAIN',
                    'analysis': '当前证据不足，无法做出明确判断。'}


class LLMAgent(BaseAgent):
    """
    LLM-powered agent using OpenAI-compatible API (GPT-4, Ollama, Qwen).
    Supports multi-round debate context injection.
    """

    def __init__(self, name: str, role: str, perspective: str,
                 api_key: str = None, model_name: str = None):
        super().__init__(name, role, perspective)
        self.api_key = api_key
        self.model_name = model_name or 'gpt-4'

    def analyze(self, context: Dict) -> Dict:
        """Send analysis prompt to LLM API. Supports debate round context."""
        prompt = self._build_prompt(context)
        response = self._call_api(prompt)
        return self._parse_response(response)

    def _build_prompt(self, context: Dict) -> str:
        """Build prompt with detection details, RAG/XAI context, and debate history."""
        detection = context.get('detection', {})
        flow_info = context.get('flow_info', {})
        rag_context = context.get('rag_context', [])
        xai_context = context.get('xai_context', {})
        round_num = context.get('round', 0)
        prev = context.get('previous_opinions', {})

        # RAG context
        rag_text = ''
        if rag_context:
            rag_text = '\n'.join(
                f"- {r.get('name', 'Unknown')} (相似度: {r.get('similarity', 0):.2f}): {r.get('description', '')}"
                for r in rag_context[:3]
            )

        # XAI context
        xai_text = ''
        if xai_context:
            top_feats = xai_context.get('top_features', [])
            if top_feats:
                xai_text = '关键特征: ' + ', '.join(
                    f"{f['name']}(重要性:{f['importance']:.4f})" for f in top_feats[:5]
                )

        # Previous debate opinions (for multi-round)
        prev_text = ''
        if round_num > 0 and prev:
            if 'detector' in prev:
                d = prev['detector']
                prev_text += f"\n🕵️ 检测员上一轮分析: {d.get('analysis', '')}\n  判决: {d.get('verdict', '')} (置信度:{d.get('confidence', 0)})\n"
            if 'analyst' in prev:
                a = prev['analyst']
                prev_text += f"\n🔬 分析师上一轮分析: {a.get('analysis', '')}\n  判决: {a.get('verdict', '')} (置信度:{a.get('confidence', 0)})\n"
            if prev_text:
                prev_text = f"\n**上一轮辩论记录:**{prev_text}\n请针对上述观点进行回应和反驳。\n"

        base = f"""你是一名网络安全分析专家，担任"{self.role}"角色。
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
{prev_text}
请从你的角色视角进行分析，**仅输出JSON格式**，不要包含其他文字:
{{"verdict": "CONFIRMED_THREAT|LIKELY_THREAT|UNCERTAIN|LIKELY_BENIGN", "confidence": 0.0-1.0, "evidence": ["证据1",...], "counter_evidence": ["反证1",...], "analysis": "详细分析"}}"""
        return base

    def _call_api(self, prompt: str) -> str:
        """Call LLM API via OpenAI-compatible endpoint. Supports Ollama JSON mode."""
        try:
            import requests
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            data = {
                'model': self.model_name,
                'messages': [
                    {'role': 'system',
                     'content': '你是网络安全分析专家，始终输出JSON格式的分析结果。不输出任何非JSON内容。'},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.2,
                'max_tokens': 800,
            }
            base_url = os.environ.get('LLM_API_BASE', 'https://api.openai.com/v1').rstrip('/')
            # Ollama 不支持 response_format, 仅在 OpenAI API 时使用
            if 'openai.com' in base_url or 'api.openai' in base_url:
                data['response_format'] = {'type': 'json_object'}
            url = f'{base_url}/chat/completions'
            resp = requests.post(url, headers=headers, json=data, timeout=60)
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
        """Parse LLM response JSON with robust fallback."""
        result = {
            'agent': self.name,
            'role': self.role,
            'verdict': 'UNCERTAIN',
            'confidence': 0.5,
            'evidence': [],
            'counter_evidence': [],
            'analysis': '',
            'llm_model': self.model_name,
        }
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                result['verdict'] = parsed.get('verdict', 'UNCERTAIN')
                result['confidence'] = float(parsed.get('confidence', 0.5))
                result['evidence'] = parsed.get('evidence', [])
                result['counter_evidence'] = parsed.get('counter_evidence', [])
                result['analysis'] = parsed.get('analysis', '')
                return result
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[LLM] Parse warning: {e}")

        # Fallback: extract from natural language
        resp_lower = response.lower()
        if 'confirmed_threat' in resp_lower or 'malicious' in resp_lower:
            result['verdict'] = 'CONFIRMED_THREAT'
        elif 'likely_threat' in resp_lower or 'suspicious' in resp_lower:
            result['verdict'] = 'LIKELY_THREAT'
        elif 'benign' in resp_lower or 'false' in resp_lower:
            result['verdict'] = 'LIKELY_BENIGN'
        result['analysis'] = response[:300]
        result['_parse_fallback'] = True
        return result


class AgentLayer:
    """
    Multi-agent debate layer.

    Three agents debate each anomaly:
    - Detector: Focuses on attack signatures and technical evidence
    - Analyst: Looks for benign explanations and false positive signals
    - Arbiter: Weighs both sides and makes final determination
    """

    def __init__(self, rag_engine=None, use_api=False, api_key=None, model_name=None,
                 debate_rounds: int = 1):
        self.rag_engine = rag_engine
        self.use_api = use_api
        self.api_key = api_key
        self.model_name = model_name
        self.debate_rounds = debate_rounds

        # Initialize agents
        if use_api and api_key:
            self._init_llm_agents()
        else:
            self._init_rule_agents()

    def _init_rule_agents(self):
        """Initialize rule-based agents with weighted rules."""
        detector_rules = [
            {'name': 'high_conf_known_attack', 'weight': 0.8, 'type': 'evidence',
             'conditions': [
                 {'metric': 'class_confidence', 'op': '>', 'value': 0.7},
                 {'metric': 'prediction', 'op': 'nin', 'value': ['BENIGN', 'UNKNOWN']},
             ], 'description': '模型高置信度判定为已知攻击类型'},
            {'name': 'unknown_with_high_score', 'weight': 0.7, 'type': 'evidence',
             'conditions': [
                 {'metric': 'is_unknown', 'op': '==', 'value': True},
                 {'metric': 'unknown_score', 'op': '>', 'value': 0.5},
             ], 'description': '开放集识别判定为未知攻击，未知概率较高'},
            {'name': 'rag_pattern_match', 'weight': 0.6, 'type': 'evidence',
             'conditions': [
                 {'metric': 'rag_has_match', 'op': '==', 'value': True},
                 {'metric': 'rag_top_similarity', 'op': '>', 'value': 0.5},
             ], 'description': 'RAG知识库匹配到高度相似的攻击模式'},
            {'name': 'anomaly_flag', 'weight': 0.5, 'type': 'evidence',
             'conditions': [
                 {'metric': 'is_anomaly', 'op': '==', 'value': True},
             ], 'description': '模型综合判定为异常流量'},
            {'name': 'high_recon_error', 'weight': 0.5, 'type': 'evidence',
             'conditions': [
                 {'metric': 'recon_error', 'op': '>', 'value': 0.1},
             ], 'description': 'CROSR重建误差较高，表明存在未见过的流量模式'},
        ]

        analyst_rules = [
            {'name': 'low_confidence', 'weight': -0.6, 'type': 'counter',
             'conditions': [
                 {'metric': 'class_confidence', 'op': '<', 'value': 0.5},
             ], 'description': '模型置信度较低，可能是误报'},
            {'name': 'predicted_benign', 'weight': -0.7, 'type': 'counter',
             'conditions': [
                 {'metric': 'prediction', 'op': '==', 'value': 'BENIGN'},
             ], 'description': '模型预测为正常流量'},
            {'name': 'no_rag_match', 'weight': -0.4, 'type': 'counter',
             'conditions': [
                 {'metric': 'rag_has_match', 'op': '==', 'value': False},
             ], 'description': 'RAG未匹配到已知攻击模式，可能是良性异常'},
            {'name': 'low_unknown_score', 'weight': -0.4, 'type': 'counter',
             'conditions': [
                 {'metric': 'unknown_score', 'op': '<', 'value': 0.3},
             ], 'description': '未知概率较低，可能仍在正常范围内'},
        ]

        arbiter_rules = [
            {'name': 'multi_evidence_match', 'weight': 0.9, 'type': 'evidence',
             'conditions': [
                 {'metric': 'class_confidence', 'op': '>', 'value': 0.8},
                 {'metric': 'rag_has_match', 'op': '==', 'value': True},
                 {'metric': 'prediction', 'op': 'nin', 'value': ['BENIGN', 'UNKNOWN']},
             ], 'description': '多维度证据一致支持威胁判定：高置信度+已知攻击+RAG匹配'},
            {'name': 'unknown_with_anomaly', 'weight': 0.7, 'type': 'evidence',
             'conditions': [
                 {'metric': 'is_unknown', 'op': '==', 'value': True},
                 {'metric': 'anomaly_score', 'op': '>', 'value': 0.5},
             ], 'description': '未知攻击且综合异常分数较高，确认威胁'},
            {'name': 'conflicting_signals', 'weight': -0.3, 'type': 'counter',
             'conditions': [
                 {'metric': 'is_unknown', 'op': '==', 'value': True},
                 {'metric': 'class_confidence', 'op': '>', 'value': 0.8},
             ], 'description': '检测信号存在矛盾：高置信度预测但被OpenMax判定为未知'},
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
        """Initialize LLM-powered agents (OpenAI/Ollama compatible)."""
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
               xai_context: Optional[Dict] = None,
               rounds: int = 1) -> Dict:
        """
        Multi-round 3-agent debate.

        Args:
            detection_result: Output from DetectionEngine.predict()
            flow_info: Network flow metadata
            rag_context: RAG search results
            xai_context: XAI explanation
            rounds: Number of debate rounds (1 = single vote, 3 = multi-round)

        Returns:
            Dict with agent opinions, voting result, final verdict
        """
        context = {
            'detection': detection_result,
            'flow_info': flow_info,
            'rag_context': rag_context or [],
            'xai_context': xai_context or {},
        }

        opinions = {}

        if rounds == 1:
            # Single round (original behavior): all three agents analyze independently
            opinions['detector'] = self.detector.analyze(context)
            opinions['analyst'] = self.analyst.analyze(context)
            opinions['arbiter'] = self.arbiter.analyze(context)
            opinions['recommendation'] = self._generate_recommendation(
                opinions['arbiter'].get('verdict', 'UNCERTAIN'),
                detection_result, flow_info, rag_context
            )
        else:
            # Multi-round: detector and analyst debate, arbiter decides last round
            for round_num in range(rounds):
                context['round'] = round_num
                context['previous_opinions'] = {
                    name: op for name, op in opinions.items()
                }
                if round_num < rounds - 1:
                    if round_num % 2 == 0:
                        opinions['detector'] = self.detector.analyze(context)
                    else:
                        opinions['analyst'] = self.analyst.analyze(context)
                else:
                    opinions['arbiter'] = self.arbiter.analyze(context)
                    opinions['recommendation'] = self._generate_recommendation(
                        opinions['arbiter'].get('verdict', 'UNCERTAIN'),
                        detection_result, flow_info, rag_context
                    )

        return self._assemble_result(opinions, detection_result, flow_info, rag_context)

    def _assemble_result(self, opinions: Dict, detection: Dict,
                          flow_info: Dict, rag_context: List[Dict]) -> Dict:
        """Assemble final debate result with weighted verdict."""
        weights = {'detector': 0.3, 'analyst': 0.3, 'arbiter': 0.4}

        def _score(verdict: str) -> float:
            return {'CONFIRMED_THREAT': 1.0, 'LIKELY_THREAT': 0.7,
                    'UNCERTAIN': 0.4, 'LIKELY_BENIGN': 0.2}.get(verdict, 0.3)

        total_weight = 0.0
        threat_score = 0.0
        for name, w in weights.items():
            if name in opinions:
                total_weight += w
                threat_score += w * _score(opinions[name].get('verdict', 'UNCERTAIN'))

        if total_weight > 0:
            threat_score /= total_weight

        # Determine final
        if threat_score >= 0.7:
            final_verdict = 'MALICIOUS'
            action = '建议立即阻断源IP，启动应急响应流程'
        elif threat_score >= 0.4:
            final_verdict = 'SUSPICIOUS'
            action = '建议持续监控该流量源，收集更多证据后研判'
        else:
            final_verdict = 'BENIGN'
            action = '疑似误报，建议标记为白名单候选'

        result = {
            'verdict': final_verdict,
            'threat_score': round(threat_score, 3),
            'action': action,
            'recommendation': opinions.get('recommendation', ''),
            'debate_mode': 'llm' if self.use_api else 'rule_based',
            'rag_context_used': bool(rag_context and len(rag_context) > 0),
            'xai_context_used': bool(detection),
            'timestamp': datetime.datetime.now().isoformat(),
        }
        # Include individual opinions
        for agent_name in ('detector', 'analyst', 'arbiter'):
            if agent_name in opinions:
                result[agent_name] = opinions[agent_name]
        return result

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
