"""
Spider-Sense v2 - Senior SecOps Analyst Agent
单安全专家智能体：融合 XAI 特征归因 + RAG 知识检索 + 检测结果，
生成结构化中文研判报告。

Supports:
- Local mode: Template-based report (no API needed)
- API mode: LLM-powered report (GPT-4/Ollama/DeepSeek/Qwen)
"""
import json
import os
import datetime
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# 特征含义映射 — 供模板报告和 LLM Prompt 共用
# ═══════════════════════════════════════════════════════════════════

FEATURE_MEANINGS = {
    'rerror_rate': ('REJ 错误率',
        '高 REJ 错误率表明连接被拒绝的比例异常，常见于暴力破解或端口扫描。'
        '在网络流量中，这表示大量连接尝试被目标拒绝。'),
    'srv_rerror_rate': ('服务 REJ 错误率',
        '同一服务的连接被拒绝率异常，表明针对特定服务的探测或攻击行为。'),
    'dst_host_rerror_rate': ('目标主机 REJ 错误率',
        '对同一目标主机的连接拒绝率异常。'),
    'dst_host_srv_rerror_rate': ('目标主机服务 REJ 错误率',
        '对同一目标主机同一服务的连接拒绝率异常，是最细粒度的错误率指标。'),
    'serror_rate': ('SYN 错误率',
        'SYN 包中发生错误的比率异常，可能表明 SYN 洪泛或网络扫描行为。'),
    'srv_serror_rate': ('服务 SYN 错误率',
        '同一服务的 SYN 错误率异常。'),
    'syn_flag_cnt': ('SYN 标志计数',
        'SYN 包数量异常高是 SYN 洪泛攻击的典型特征。'
        '攻击者发送大量 SYN 包但不完成三次握手，耗尽目标半开连接队列。'),
    'ack_flag_cnt': ('ACK 标志计数',
        'ACK 包比例异常，大量 ACK 包可能表明 ACK 洪泛攻击。'),
    'fin_flag_cnt': ('FIN 标志计数',
        'FIN 包数量异常，可能是 FIN 扫描或连接终止攻击。'),
    'rst_flag_cnt': ('RST 标志计数',
        'RST 包数量异常，常见于端口扫描或拒绝服务攻击中强制断开连接。'),
    'psh_flag_cnt': ('PSH 标志计数',
        'PSH 标志计数异常，推送标志的异常使用模式。'),
    'urg_flag_cnt': ('URG 标志计数',
        'URG 标志异常使用，某些攻击利用紧急指针进行探测。'),
    'flow_pkts_s': ('包速率 (packets/s)',
        '每秒数据包数量。异常高的包速率是 DDoS 洪泛攻击的核心特征。'),
    'flow_byts_s': ('字节速率 (bytes/s)',
        '每秒字节数。DDoS 攻击通常伴随极高的带宽消耗。'),
    'flow_duration': ('流持续时间',
        '网络流的持续时间。极短或极长的连接可能是扫描或慢速攻击的特征。'),
    'fwd_pkt_len_mean': ('前向包平均长度',
        '前向数据包平均大小。DDoS 攻击中包大小通常均匀，正常流量中变化较大。'),
    'fwd_pkt_len_std': ('前向包长度标准差',
        '前向数据包大小的变异程度。标准差极低（包大小均匀）是自动化攻击工具的特征。'),
    'pkt_len_var': ('包长度方差',
        '所有数据包长度的方差。均匀包大小（低方差）表明自动化工具生成，高方差是正常交互。'),
    'pkt_len_std': ('包长度标准差',
        '包长度标准差。与 pkt_len_var 类似，低标准差暗示自动化攻击。'),
    'down_up_ratio': ('下行/上行比率',
        '下载与上传数据量之比。DDoS 攻击常表现为极端不对称（几乎只有入站流量）。'),
    'init_win_bytes_fwd': ('前向 TCP 窗口大小',
        'TCP 初始窗口大小。某些恶意软件使用异常的窗口大小进行通信。'),
    'init_win_bytes_bwd': ('后向 TCP 窗口大小',
        '反向 TCP 初始窗口大小。'),
    'subflow_fwd_pkts': ('子流前向包数',
        '子流中前向数据包数量。'),
    'subflow_bwd_pkts': ('子流后向包数',
        '子流中后向数据包数量。'),
    'active_mean': ('平均活跃时间',
        '连接处于活跃状态的平均时长。慢速攻击的特征是连接保持活跃但数据量极小。'),
    'idle_mean': ('平均空闲时间',
        '连接空闲的平均时长。异常长或异常短的空闲模式可能表明 C2 信标行为。'),
    'fwd_iat_mean': ('前向包到达间隔均值',
        '前向数据包到达的平均时间间隔。规律性间隔可能是 C2 信标；极高频率是 DDoS。'),
    'bwd_iat_mean': ('后向包到达间隔均值',
        '后向数据包到达的平均时间间隔。'),
    'fwd_iat_std': ('前向包到达间隔标准差',
        '前向数据包间隔的规律性。低标准差=规律性/自动化；高标准差=人类行为。'),
    'dst_host_count': ('目标主机连接数',
        '与同一目标 IP 建立的连接数。异常高的连接数可能是 DDoS 或暴力破解。'),
    'dst_host_srv_count': ('目标主机服务连接数',
        '对同一目标主机同一服务的连接数。细粒度指标，精确反映针对特定服务的攻击。'),
    'count': ('总连接数',
        '短时间内建立的总连接数。端口扫描的特征是该数字急剧增加。'),
    'srv_count': ('服务连接数',
        '同一服务的连接数。暴力破解通常对同一服务发起大量连接。'),
    'same_srv_rate': ('同服务连接比例',
        '连接到同一服务的比例。攻击通常集中在单一服务上。'),
    'diff_srv_rate': ('不同服务连接比例',
        '连接到不同服务的比例。扫描行为该值较低。'),
    'dst_host_same_srv_rate': ('同目标同服务比例',
        '对同一目标同一服务的连接比例。'),
    'dst_host_diff_srv_rate': ('同目标不同服务比例',
        '对同一目标不同服务的连接比例。横向扫描的特征。'),
    'fwd_pkts_s': ('前向包速率',
        '每秒前向数据包数。'),
    'bwd_pkts_s': ('后向包速率',
        '每秒后向数据包数。'),
    'flow_iat_mean': ('流到达间隔均值',
        '流级别的到达间隔。'),
    'fwd_header_length': ('前向头部长度',
        '前向 IP/TCP 头部长度。异常头部长度可能是协议异常或隧道行为。'),
    'average_packet_size': ('平均包大小',
        '所有数据包的平均大小。'),
    'min_seg_size_forward': ('前向最小分段大小',
        '前向 TCP 分段的最小值。'),
}


# ═══════════════════════════════════════════════════════════════════
# Senior SecOps Analyst Agent
# ═══════════════════════════════════════════════════════════════════

class SeniorSecOpsAgent:
    """
    单一安全专家 Agent — 替代原来的三智能体辩论。

    输入 (由 Orchestrator 汇总):
      - detection:  OpenMax 检测结果 (prediction, unknown_score, class_confidence, ...)
      - flow_info: 流量元数据 (src_ip, dst_ip, protocol, ...)
      - xai:       XAI 扰动归因 Top-N 特征列表
      - rag:       RAG 语义检索命中的 MITRE 知识条目

    输出:
      {
        'verdict': 'MALICIOUS' | 'SUSPICIOUS' | 'BENIGN',
        'risk_assessment': '...',      # 风险评估
        'intelligence_correlation': '...',  # 情报关联
        'disposal_recommendation': '...',   # 处置建议
        'threat_score': 0.0-1.0,
        'timestamp': '...',
      }
    """

    def __init__(self, use_api: bool = False, api_key: str = None,
                 model_name: str = 'gpt-4'):
        self.use_api = use_api and bool(api_key)
        self.api_key = api_key
        self.model_name = model_name or 'gpt-4'

    # ── 主入口 ─────────────────────────────────────────────────────

    def analyze(self,
                detection: Dict,
                flow_info: Dict,
                xai_result: Optional[Dict] = None,
                rag_result: Optional[List[Dict]] = None) -> Dict:
        """生成结构化研判报告。"""
        if self.use_api:
            return self._llm_report(detection, flow_info, xai_result, rag_result)
        else:
            return self._template_report(detection, flow_info, xai_result, rag_result)

    # ── 离线模板报告 (无需 LLM) ─────────────────────────────────────

    def _template_report(self, detection, flow_info, xai, rag) -> Dict:
        """基于 XAI 真实数据 + RAG 知识库的离线报告模板。"""

        prediction = detection.get('prediction', 'N/A')
        is_unknown = detection.get('is_unknown', False)
        unknown_score = detection.get('unknown_score', 0)
        class_conf = detection.get('class_confidence', 0)
        anomaly_score = detection.get('anomaly_score', 0)
        recon_error = detection.get('recon_error', 0)

        # ── 1. 风险评估 ──────────────────────────────────────────
        risk_assessment_parts = []

        if is_unknown:
            risk_assessment_parts.append(
                f"OpenMax 开放集识别判定该流量为未知攻击类型，"
                f"未知概率 {unknown_score:.0%}。"
                f"该流量的特征模式与 6 种已知攻击类别的 Weibull 分布均不匹配，"
                f"表明可能为零日攻击或已知攻击的显著变种。"
            )
        else:
            risk_assessment_parts.append(
                f"模型判定为 {prediction}，分类置信度 {class_conf:.0%}。"
            )

        # 重建误差
        if recon_error > 0.15:
            risk_assessment_parts.append(
                f"CROSR 重建误差 {recon_error:.4f} 超过阈值 0.15，"
                f"表明 decoder 无法有效重建该输入，流量模式偏离训练分布。"
            )
        else:
            risk_assessment_parts.append(
                f"CROSR 重建误差 {recon_error:.4f} 在正常范围内。"
            )

        # XAI 特征归因
        if xai and xai.get('top_features'):
            top_feats = xai['top_features'][:5]
            feat_descs = []
            for f in top_feats:
                # Map common feature names to Chinese explanations
                desc = self._explain_feature(f['name'], f['value'],
                                             f['importance'], detection)
                feat_descs.append(desc)
            risk_assessment_parts.append("关键特征归因分析: " + "；".join(feat_descs))
        else:
            risk_assessment_parts.append("无显著特征归因可用。")

        # 综合异常分数
        risk_assessment_parts.append(
            f"综合异常分数: {anomaly_score:.4f} "
            f"(由 unknown_score={unknown_score:.4f}, "
            f"1-class_conf={1.0-class_conf:.4f}, "
            f"recon_error={recon_error:.4f} 加权合成)。"
        )

        risk_assessment = "\n".join(risk_assessment_parts)

        # ── 2. 情报关联 ──────────────────────────────────────────
        if rag:
            intel_parts = []
            for r in rag[:3]:
                intel_parts.append(
                    f"• [{r.get('name', 'N/A')}] (相似度 {r.get('similarity', 0):.2f})"
                )
                if r.get('description'):
                    intel_parts.append(f"  {r['description']}")
                mitre = r.get('mitre')
                if mitre and mitre.get('id'):
                    intel_parts.append(
                        f"  MITRE {mitre['id']}: {mitre.get('name', '')} "
                        f"| 战术: {', '.join(mitre.get('tactics', []))}"
                    )
            intelligence_correlation = (
                f"RAG 知识库检索命中 {len(rag)} 条相关攻击模式:\n"
                + "\n".join(intel_parts)
            )
        else:
            intelligence_correlation = "RAG 知识库未匹配到高度相关的攻击模式，"
            intelligence_correlation += "建议提交至安全分析师进行人工研判。"

        # ── 3. 处置建议 ──────────────────────────────────────────
        disposal = self._generate_disposal(detection, flow_info, rag)

        # ── 4. 威胁评级 ──────────────────────────────────────────
        threat_score, verdict = self._compute_threat(detection, xai, rag)

        return {
            'verdict': verdict,
            'threat_score': round(threat_score, 3),
            'risk_assessment': risk_assessment,
            'intelligence_correlation': intelligence_correlation,
            'disposal_recommendation': disposal,
            'analysis_mode': 'template',
            'timestamp': datetime.datetime.now().isoformat(),
        }

    def _get_feature_meaning(self, name: str) -> tuple:
        """返回 (中文名, 安全含义解释)。供 LLM prompt 和模板报告共用。"""
        return FEATURE_MEANINGS.get(name, (name, f'{name} 特征，具体含义待补充'))

    def _explain_feature(self, name: str, value: float, importance: float,
                         detection: Dict) -> str:
        """将单个特征的扰动归因结果翻译为中文安全解释。"""
        display_name, explanation = self._get_feature_meaning(name)

        # Normalize importance for display
        imp_pct = importance * 100

        desc = f"{display_name}({name}) 贡献度 {imp_pct:.1f}%"
        if explanation:
            desc += f"，{explanation}"
        return desc

    def _generate_disposal(self, detection, flow_info, rag) -> str:
        """根据检测结果和 RAG 知识生成处置建议。"""
        is_unknown = detection.get('is_unknown', False)
        unknown_score = detection.get('unknown_score', 0)
        prediction = detection.get('prediction', 'N/A')
        src_ip = flow_info.get('src_ip', '未知来源')

        lines = []

        # 提取 RAG 中的缓解措施
        mitigations = []
        if rag:
            for r in rag[:3]:
                mitre = r.get('mitre')
                if mitre and mitre.get('id'):
                    # 从 MITRE 知识中提取关键词
                    desc = mitre.get('description', '')
                    if '流量清洗' in desc:
                        mitigations.append('流量清洗/抗DDoS服务')
                    if '速率限制' in desc:
                        mitigations.append('速率限制(Rate Limiting)')
                    if '网络分段' in desc:
                        mitigations.append('网络分段隔离')
                    if '防火墙' in desc:
                        mitigations.append('配置防火墙规则')
                    if '多因素认证' in desc:
                        mitigations.append('启用多因素认证(MFA)')
                    if 'SYN Cookie' in desc:
                        mitigations.append('启用 SYN Cookie')
                    if '账户锁定' in desc:
                        mitigations.append('配置账户锁定策略')
        mitigations = list(set(mitigations))  # 去重

        if is_unknown and unknown_score > 0.5:
            lines.append(f"🚨 该流量被判定为高危未知攻击 (未知概率 {unknown_score:.0%})。")
            lines.append(f"建议立即采取以下措施:")
            lines.append(f"  1. 临时封锁源 IP {src_ip}，防止潜在攻击扩散")
            lines.append(f"  2. 捕获完整 PCAP 数据包，提交至安全实验室进行深度分析")
            lines.append(f"  3. 检查目标主机是否出现异常行为（CPU/内存/网络连接数）")
            lines.append(f"  4. 将流量特征提交至威胁情报平台，查询是否匹配已知新型攻击")
        elif is_unknown:
            lines.append(f"⚠️ 该流量被判定为中低风险未知异常 (未知概率 {unknown_score:.0%})。")
            lines.append(f"建议:")
            lines.append(f"  1. 对源 IP {src_ip} 启用增强监控，持续观察")
            lines.append(f"  2. 关联其他安全日志（IDS/防火墙/端点）进行综合研判")
            lines.append(f"  3. 如持续出现类似模式，升级处置级别")
        elif detection.get('anomaly_type') == 'known_attack':
            lines.append(f"🎯 该流量被判定为已知攻击: {prediction}。")
            lines.append(f"建议:")
            lines.append(f"  1. 立即阻断源 IP {src_ip}")
            lines.append(f"  2. 检查目标主机是否已被入侵")
            lines.append(f"  3. 生成安全事件工单，启动应急响应流程")

        if mitigations:
            lines.append(f"\n技术缓解措施 (来自 MITRE ATT&CK):")
            for m in mitigations[:5]:
                lines.append(f"  • {m}")

        if not mitigations and not is_unknown:
            lines.append(f"\n暂无自动匹配的 MITRE 缓解措施，建议人工研判后确定处置方案。")

        return "\n".join(lines)

    def _compute_threat(self, detection, xai, rag) -> tuple:
        """综合 OpenMax + XAI + RAG 计算威胁分数和评级。"""
        unknown_score = detection.get('unknown_score', 0)
        anomaly_score = detection.get('anomaly_score', 0)
        is_unknown = detection.get('is_unknown', False)
        recon_error = detection.get('recon_error', 0)

        # 基础威胁分数 = 异常分数
        score = anomaly_score

        # XAI 归因增强: 如果有显著的特征归因 (importance > 0.05), 增强置信度
        if xai and xai.get('top_features'):
            significant = [f for f in xai['top_features'] if f['importance'] > 0.05]
            if len(significant) >= 3:
                score = min(1.0, score + 0.1)

        # RAG 匹配增强: 如果有高相似度匹配, 增强置信度
        if rag:
            high_sim = [r for r in rag if r.get('similarity', 0) > 0.5]
            if high_sim:
                score = min(1.0, score + 0.1)

        # 重建误差增强
        if recon_error > 0.15:
            score = min(1.0, score + 0.05)

        # 定级
        if score >= 0.7:
            verdict = 'MALICIOUS'
        elif score >= 0.4:
            verdict = 'SUSPICIOUS'
        else:
            verdict = 'BENIGN'

        return score, verdict

    # ── LLM 增强报告 (有 API Key 时) ─────────────────────────────────

    def _llm_report(self, detection, flow_info, xai, rag) -> Dict:
        """调用 LLM 生成专业中文研判报告。失败时自动降级为模板。"""
        try:
            prompt = self._build_llm_prompt(detection, flow_info, xai, rag)
            response = self._call_api(prompt)
            if response and response != '{}':
                return self._parse_llm_response(response, detection)
        except Exception as e:
            print(f"[Agent] LLM 调用异常, 降级为模板: {e}")

        # 降级：用模板生成，但标记为 llm_fallback
        result = self._template_report(detection, flow_info, xai, rag)
        result['analysis_mode'] = 'llm_fallback'
        result['llm_model'] = self.model_name
        return result

    def _build_llm_prompt(self, detection, flow_info, xai, rag) -> str:
        """构造 LLM prompt，关键: XAI 数据是真实算出来的，LLM 只负责解释。"""
        prediction = detection.get('prediction', 'N/A')
        is_unknown = detection.get('is_unknown', False)
        unknown_score = detection.get('unknown_score', 0)
        class_conf = detection.get('class_confidence', 0)
        anomaly_score = detection.get('anomaly_score', 0)
        recon_error = detection.get('recon_error', 0)

        # XAI 特征归因 (真实数据 + 中文含义解释)
        xai_text = ''
        if xai and xai.get('top_features'):
            for f in xai['top_features'][:8]:
                display_name, explanation = self._get_feature_meaning(f['name'])
                xai_text += (
                    f"  - {f['name']}（{display_name}）: "
                    f"扰动归因贡献度 {f['importance']:.4f}, "
                    f"原始值 {f['value']:.4f}。"
                    f"安全含义: {explanation}\n"
                )
        else:
            xai_text = '暂无可用的特征归因数据。'

        # RAG 知识 (真实检索结果)
        rag_text = ''
        if rag:
            rag_text = '\n'.join(
                f"  - [{r.get('name', 'N/A')}] 相似度 {r.get('similarity', 0):.2f}\n"
                f"    描述: {r.get('description', '无')}\n"
                f"    MITRE: {r.get('mitre', {}).get('id', 'N/A')} "
                f"{r.get('mitre', {}).get('name', '')} "
                f"| 战术: {', '.join(r.get('mitre', {}).get('tactics', []))}"
                for r in (rag or [])[:3]
            )
        else:
            rag_text = '未匹配到相关的 MITRE ATT&CK 技术条目。'

        prompt = f"""你是一名资深的 SOC（安全运营中心）高级分析师。系统拦截到一笔可疑网络流量，
请结合检测器输出的数学数据、XAI 特征归因结果和 RAG 检索到的威胁情报，
输出一份中文研判报告。

⚠️ 重要约束: XAI 特征归因数据是由扰动法在实际模型上真实计算的数学结果，
你只能基于这些数据进行安全语义解释，不得编造或修改归因数值。

【检测器告警】
- 预测类别: {prediction}
- 是否为未知攻击: {is_unknown}
- 未知攻击概率 (OpenMax): {unknown_score:.2%}
- 已知类置信度 (OpenMax): {class_conf:.2%}
- 综合异常分数: {anomaly_score:.4f}
- CROSR 重建误差: {recon_error:.6f}

【流量元数据】
- 源 IP: {flow_info.get('src_ip', 'N/A')}
- 目标 IP: {flow_info.get('dst_ip', 'N/A')}
- 协议: {flow_info.get('protocol', 'N/A')}
- 包长度: {flow_info.get('packet_length', 0)}

【XAI 特征归因 (数学计算, 请勿修改数值)】
{xai_text}

【关联威胁情报 (RAG 检索)】
{rag_text}

请严格按照以下格式输出 JSON (不要包含任何其他文字):

{{
  "risk_assessment": "详细的风险评估，结合 XAI 特征归因中的关键特征进行安全原理解释。例如: rerror_rate 高说明连接被拒绝的比例异常，常见于暴力破解；syn_flag_cnt 异常高说明 SYN 包比例异常，符合 SYN 洪泛攻击特征。必须基于上面给出的 XAI 真实数值进行分析，不可编造。",
  "intelligence_correlation": "RAG 检索到的 MITRE ATT&CK 技术是否与当前检测结果吻合？如果不吻合，请说明可能的原因。如果吻合，请解释攻击链中各阶段的对应关系。",
  "disposal_recommendation": "面向运维人员的可操作处置建议，分优先级列出。引用 RAG 中 MITRE 技术的具体缓解措施。",
  "verdict": "MALICIOUS | SUSPICIOUS | BENIGN",
  "threat_score": 0.0
}}"""

        return prompt

    def _call_api(self, prompt: str) -> str:
        """Call LLM API via OpenAI-compatible endpoint."""
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
                     'content': '你是资深 SOC 安全分析师，始终严格按 JSON 格式输出研判报告。不输出任何非 JSON 内容。'},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.3,
                'max_tokens': 1200,
            }
            base_url = os.environ.get('LLM_API_BASE', 'https://api.openai.com/v1').rstrip('/')
            if 'openai.com' in base_url or 'api.openai' in base_url:
                data['response_format'] = {'type': 'json_object'}
            url = f'{base_url}/chat/completions'
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            body = resp.text[:512]
            print(f"[LLM] {resp.status_code}: {body}")
            return '{}'
        except Exception as e:
            print(f"[LLM] API error: {e}")
            return '{}'

    def _parse_llm_response(self, response: str, detection: Dict) -> Dict:
        """Parse LLM JSON response with fallback to template."""
        result = {
            'verdict': 'SUSPICIOUS',
            'threat_score': 0.5,
            'risk_assessment': '',
            'intelligence_correlation': '',
            'disposal_recommendation': '',
            'analysis_mode': 'llm',
            'llm_model': self.model_name,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                result['verdict'] = parsed.get('verdict', 'SUSPICIOUS')
                result['threat_score'] = float(parsed.get('threat_score', 0.5))
                result['risk_assessment'] = parsed.get('risk_assessment', '')
                result['intelligence_correlation'] = parsed.get('intelligence_correlation', '')
                result['disposal_recommendation'] = parsed.get('disposal_recommendation', '')
                return result
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[LLM] Parse error: {e}")

        # Fallback: use raw response as analysis
        result['risk_assessment'] = response[:500]
        result['analysis_mode'] = 'llm_fallback'
        return result


# ═══════════════════════════════════════════════════════════════════
# AgentLayer — 兼容旧接口的薄封装
# ═══════════════════════════════════════════════════════════════════

class AgentLayer:
    """
    兼容层: 保持与 orchestrator 和 app.py 的接口一致，
    内部已切换为 SeniorSecOpsAgent 单智能体模式。
    """

    def __init__(self, rag_engine=None, use_api=False, api_key=None,
                 model_name=None, **kwargs):
        self.rag_engine = rag_engine
        self.use_api = use_api and bool(api_key)
        self.api_key = api_key
        self.model_name = model_name or 'gpt-4'

        self.agent = SeniorSecOpsAgent(
            use_api=self.use_api,
            api_key=api_key,
            model_name=self.model_name,
        )

    def debate(self, detection_result: Dict, flow_info: Dict,
               rag_context: Optional[List[Dict]] = None,
               xai_context: Optional[Dict] = None,
               rounds: int = 1) -> Dict:
        """
        兼容旧 debate() 接口 — 内部转发到单 Agent。
        rounds 参数保留但忽略（单 Agent 不需要多轮辩论）。
        """
        return self.agent.analyze(
            detection=detection_result,
            flow_info=flow_info,
            xai_result=xai_context,
            rag_result=rag_context,
        )
