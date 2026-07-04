"""
Spider-Sense v2 - Semantic RAG Engine
Flow-to-text → SentenceTransformer embedding → MITRE ATT&CK semantic search.

原理:
  1. FlowToText 将 78 维流特征转为英语行为描述
  2. all-MiniLM-L6-v2 编码为 384 维语义向量
  3. 与预计算的 MITRE ATT&CK key_text embedding 做余弦相似度匹配
  4. 返回 Top-K 最相关的攻击技术 (含 description, detection, mitigation)

依赖: pip install sentence-transformers
"""
import json
import os
import numpy as np
from typing import Dict, List, Optional


class SemanticRAG:
    """
    Semantic RAG — 流行为描述 → MITRE ATT&CK 知识检索。

    使用方式:
        rag = SemanticRAG(knowledge_dir='knowledge')
        rag.initialize()  # 加载 MITRE + 预计算 embedding (首次较慢)

        # 检索
        results = rag.search("High packet rate on port 80, SYN flood signature...")
        # → [{'technique_id': 'T1498', 'similarity': 0.87, ...}, ...]
    """

    def __init__(self, knowledge_dir: str = 'knowledge',
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.knowledge_dir = knowledge_dir
        self.model_name = model_name
        self.embedder = None
        self.knowledge_entries = []   # list of {id, key_text, value, ...}
        self.doc_embeddings = None    # [N, 384] numpy array
        self._initialized = False

    # ── 初始化 ─────────────────────────────────────────────────────

    def initialize(self, mitre_json_path: str = None) -> 'SemanticRAG':
        """
        加载 MITRE 知识库，预计算所有 key_text 的 embedding。
        首次运行需下载模型 (~90MB)，后续启动直接使用缓存。
        """
        # 1. 加载 embedding 模型 (自动缓存到 ~/.cache/huggingface)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装。请运行: pip install sentence-transformers"
            )

        # 2. 加载 MITRE 知识库 (优先完整版, 回退手写版)
        if mitre_json_path is None:
            # 优先: 从 enterprise-attack.json 生成的完整知识库
            full_path = os.path.join(self.knowledge_dir, 'mitre_semantic.json')
            basic_path = os.path.join(self.knowledge_dir, 'mitre_attack.json')
            if os.path.exists(full_path):
                mitre_json_path = full_path
            elif os.path.exists(basic_path):
                mitre_json_path = basic_path

        if mitre_json_path and os.path.exists(mitre_json_path):
            entries = self._load_mitre_entries(mitre_json_path)
        else:
            print(f"[SemanticRAG] No MITRE JSON found, using minimal fallback")
            entries = self._fallback_entries()

        self.knowledge_entries = entries

        # 追加正常流量锚点 — 防止 BENIGN 被 RAG 误匹配到攻击技术
        benign_entries = self._benign_anchors()
        self.knowledge_entries.extend(benign_entries)
        print(f"[SemanticRAG] Loaded {len(entries)} MITRE + "
              f"{len(benign_entries)} benign anchors = {len(self.knowledge_entries)} total entries")

        # 3. 预计算 embedding
        self._build_index()
        self._initialized = True
        return self

    def _load_mitre_entries(self, json_path: str) -> List[Dict]:
        """
        加载 MITRE 知识条目。兼容两种格式:
          - 新格式 (mitre_semantic.json): [{"id":..., "key_text":..., "value":{...}}, ...]
          - 旧格式 (mitre_attack.json):  {tech_id: {name, tactics, ...}, ...}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = []

        # 检测格式: list → 新格式, dict → 旧格式
        if isinstance(data, list):
            # 新格式: 直接使用预构建的 key_text + value
            for item in data:
                entries.append({
                    'id': item.get('id', ''),
                    'key_text': item.get('key_text', ''),
                    'value': item.get('value', {}),
                    'type': item.get('type', 'mitre_technique'),
                })
            print(f"[SemanticRAG] Detected new format (mitre_semantic.json): "
                  f"{len(entries)} pre-built entries")

        elif isinstance(data, dict):
            # 旧格式: 手写的 {tech_id: {...}}, 需要自己拼 key_text
            for tech_id, tech in data.items():
                tactics = ', '.join(tech.get('tactics', []))
                platforms = ', '.join(tech.get('platforms', []))

                key_text = (
                    f"MITRE ATT&CK technique {tech_id}: {tech.get('name', '')}. "
                    f"Tactics: {tactics}. "
                    f"Platforms: {platforms}. "
                    f"Description: {tech.get('description', '')}. "
                    f"Detection: {tech.get('detection', '')}. "
                    f"Mitigation: {tech.get('mitigation', '')}."
                )

                entries.append({
                    'id': tech_id,
                    'key_text': key_text,
                    'value': {
                        'technique_id': tech_id,
                        'technique_name': tech.get('name', ''),
                        'tactics': tech.get('tactics', []),
                        'platforms': tech.get('platforms', []),
                        'description': tech.get('description', ''),
                        'detection': tech.get('detection', ''),
                        'mitigation': tech.get('mitigation', ''),
                    },
                    'type': 'mitre_technique',
                })
            print(f"[SemanticRAG] Detected old format (mitre_attack.json): "
                  f"{len(entries)} dict entries")

        return entries

    def _fallback_entries(self) -> List[Dict]:
        """Fallback: 内嵌最小 MITRE 知识集。"""
        entries = []
        defaults = [
            ('T1498', 'Network Denial of Service', ['Impact'],
             'Adversaries perform network DoS by exhausting bandwidth. '
             'Observable: extreme packet rate, SYN flood, uniform packet sizes, '
             'high inbound traffic volume. Detection: baseline deviation, '
             'SYN ratio monitoring. Mitigation: traffic scrubbing, rate limiting, CDN.'),
            ('T1499.002', 'Service Exhaustion Flood', ['Impact'],
             'Adversaries target application-layer resources. '
             'Observable: many concurrent connections, low throughput per connection, '
             'HTTP 503 errors, connection pool saturation. '
             'Detection: error rate spike, queue depth anomaly. '
             'Mitigation: connection timeout tuning, per-IP connection limits, reverse proxy.'),
            ('T1046', 'Network Service Scanning', ['Discovery'],
             'Adversaries probe ports to discover services. '
             'Observable: multiple ports contacted from single source in short window, '
             'SYN scan patterns, sequential port access, tiny packets. '
             'Detection: port scan signature, connection rate anomaly. '
             'Mitigation: network segmentation, firewall rules, port knocking.'),
            ('T1110', 'Brute Force', ['Credential Access'],
             'Adversaries use brute force to guess credentials. '
             'Observable: repeated authentication failures, high login rate, '
             'same service targeted from single IP. '
             'Detection: login failure rate, account lockout events. '
             'Mitigation: account lockout, MFA, rate limiting on auth endpoints.'),
            ('T1573', 'Encrypted Channel', ['Command and Control'],
             'Adversaries use encrypted channels for C2. '
             'Observable: abnormal TLS certificates, non-standard ports, '
             'periodic beaconing, JA3 hash anomalies. '
             'Detection: TLS fingerprinting, certificate analysis. '
             'Mitigation: SSL/TLS decryption proxy, certificate pinning.'),
            ('T1071', 'Application Layer Protocol', ['Command and Control'],
             'Adversaries use app-layer protocols for C2. '
             'Observable: periodic HTTP/HTTPS/DNS connections, beaconing intervals, '
             'unusual User-Agent strings. '
             'Detection: beacon analysis, DNS tunneling detection. '
             'Mitigation: DNS security, app-layer firewall, HTTPS inspection.'),
            ('T1041', 'Exfiltration Over C2 Channel', ['Exfiltration'],
             'Adversaries exfiltrate data over C2. '
             'Observable: large outbound transfers, non-standard ports, '
             'off-hours activity, asymmetric traffic. '
             'Detection: outbound traffic anomaly, DLP monitoring. '
             'Mitigation: DLP, egress filtering, network segmentation.'),
            ('T1190', 'Exploit Public-Facing Application', ['Initial Access'],
             'Adversaries exploit vulnerabilities in internet-facing apps. '
             'Observable: unusual HTTP request patterns, known exploit payloads. '
             'Detection: WAF log analysis, vulnerability scanning. '
             'Mitigation: regular patching, WAF, least privilege.'),
        ]
        for tech_id, name, tactics, text in defaults:
            entries.append({
                'id': tech_id,
                'key_text': f"MITRE ATT&CK {tech_id}: {name}. Tactics: {', '.join(tactics)}. {text}",
                'value': {
                    'technique_id': tech_id, 'technique_name': name,
                    'tactics': tactics, 'description': text,
                    'detection': '', 'mitigation': '',
                },
                'type': 'mitre_technique',
            })
        return entries

    def _benign_anchors(self) -> List[Dict]:
        """
        正常流量锚点条目。

        问题: MITRE 知识库 858 条全是攻击技术。当正常流量被 OpenMax 误判为
        UNKNOWN 时，RAG 在所有攻击里找个最像的 → 强制误匹配。

        解决: 在知识库中插入正常流量行为描述。当流量的行为模式接近这些描述时，
        RAG 会匹配到 'BENIGN' 而非攻击技术，避免误报。
        """
        anchors = [
            {
                'id': 'BENIGN_DNS',
                'key_text': (
                    'Normal DNS query traffic. Observable: destination port 53 UDP, '
                    'small packet sizes typically under 512 bytes, short duration '
                    'under 1 second, single query-response exchange, low bandwidth, '
                    'one or few packets per flow. NOT consistent with DNS tunneling, '
                    'DNS amplification, or DNS exfiltration. This is legitimate '
                    'domain name resolution behavior.'
                ),
                'value': {
                    'technique_id': 'BENIGN', 'technique_name': 'Normal DNS Query',
                    'tactics': [], 'tactics_cn': '正常流量',
                    'description': '标准 DNS 域名解析流量。短连接、小包体、低带宽。',
                    'detection': '', 'mitigation': '',
                },
                'type': 'benign_anchor',
            },
            {
                'id': 'BENIGN_HTTP',
                'key_text': (
                    'Normal HTTP web browsing traffic. Observable: destination port 80 '
                    'or 8080, variable packet sizes reflecting mixed content (HTML, CSS, '
                    'images), moderate bandwidth, moderate duration, request-response '
                    'pattern, human-driven timing with irregular intervals. NOT automated '
                    'tool traffic, NOT DDoS flood, NOT scanning. Typical web browsing '
                    'with variable payload sizes driven by user interaction.'
                ),
                'value': {
                    'technique_id': 'BENIGN', 'technique_name': 'Normal HTTP Browsing',
                    'tactics': [], 'tactics_cn': '正常流量',
                    'description': '标准 HTTP 网页浏览流量。包大小变化大、人工交互节奏。',
                    'detection': '', 'mitigation': '',
                },
                'type': 'benign_anchor',
            },
            {
                'id': 'BENIGN_HTTPS',
                'key_text': (
                    'Normal HTTPS encrypted web traffic. Observable: destination port 443, '
                    'encrypted payload, TLS handshake completed, variable but moderate '
                    'bandwidth, human interactive timing patterns, standard TLS certificate '
                    'from trusted CA, normal JA3 fingerprint. NOT self-signed certificate, '
                    'NOT C2 beaconing, NOT data exfiltration. Regular encrypted web browsing.'
                ),
                'value': {
                    'technique_id': 'BENIGN', 'technique_name': 'Normal HTTPS Browsing',
                    'tactics': [], 'tactics_cn': '正常流量',
                    'description': '标准 HTTPS 加密网页浏览。合法证书、人工交互节奏。',
                    'detection': '', 'mitigation': '',
                },
                'type': 'benign_anchor',
            },
            {
                'id': 'BENIGN_FILE_TRANSFER',
                'key_text': (
                    'Normal file transfer or software update traffic. Observable: moderate '
                    'to high bandwidth, large packet sizes, long duration, bulk data '
                    'movement, standard ports (FTP 21, or HTTP/HTTPS for downloads), '
                    'clean connection teardown with FIN handshake. NOT C2 download, '
                    'NOT malware dropper, NOT data exfiltration pattern. Legitimate '
                    'bulk data transfer with normal protocol behavior.'
                ),
                'value': {
                    'technique_id': 'BENIGN', 'technique_name': 'Normal File Transfer',
                    'tactics': [], 'tactics_cn': '正常流量',
                    'description': '正常文件传输/软件更新。大流量、长连接、干净关闭。',
                    'detection': '', 'mitigation': '',
                },
                'type': 'benign_anchor',
            },
            {
                'id': 'BENIGN_DATABASE',
                'key_text': (
                    'Normal database client-server traffic. Observable: destination port '
                    '3306 MySQL, 5432 PostgreSQL, 1433 MSSQL, or 27017 MongoDB, '
                    'structured query-response pattern, moderate packet sizes, '
                    'persistent or semi-persistent connections, moderate bandwidth, '
                    'internal network source and destination. NOT SQL injection, '
                    'NOT database exploitation, NOT data exfiltration. Standard '
                    'database operations within normal business context.'
                ),
                'value': {
                    'technique_id': 'BENIGN', 'technique_name': 'Normal Database Query',
                    'tactics': [], 'tactics_cn': '正常流量',
                    'description': '标准数据库客户端-服务器交互。结构化查询-响应模式。',
                    'detection': '', 'mitigation': '',
                },
                'type': 'benign_anchor',
            },
            {
                'id': 'BENIGN_LOW_VOLUME',
                'key_text': (
                    'Low volume background or idle traffic. Observable: very low packet '
                    'rate under 1 pkts/s, minimal bandwidth, keepalive or heartbeat '
                    'packets, long idle periods, small packet sizes. NOT C2 beaconing '
                    '(no periodic regularity), NOT slow attack (no connection accumulation), '
                    'NOT scanning (single destination, no port diversity). Normal '
                    'background noise and connection maintenance traffic.'
                ),
                'value': {
                    'technique_id': 'BENIGN', 'technique_name': 'Normal Background Traffic',
                    'tactics': [], 'tactics_cn': '正常流量',
                    'description': '低量背景/空闲流量。keepalive 心跳、无攻击模式。',
                    'detection': '', 'mitigation': '',
                },
                'type': 'benign_anchor',
            },
        ]
        return anchors

    def _build_index(self):
        """预计算所有 key_text 的 embedding（首次慢，后续从缓存秒加载）。"""
        cache_path = os.path.join(self.knowledge_dir, 'mitre_embeddings.npy')
        hash_path = os.path.join(self.knowledge_dir, 'mitre_embeddings.hash')

        # 检测 MITRE JSON 是否变化（用文件大小+修改时间做简易 hash）
        json_path = os.path.join(self.knowledge_dir,
                                 'mitre_semantic.json' if os.path.exists(
                                     os.path.join(self.knowledge_dir, 'mitre_semantic.json'))
                                 else 'mitre_attack.json')
        json_hash = str(os.path.getmtime(json_path)) + '_' + str(os.path.getsize(json_path))

        # 尝试从缓存加载
        if os.path.exists(cache_path) and os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                cached_hash = f.read().strip()
            if cached_hash == json_hash:
                self.doc_embeddings = np.load(cache_path)
                print(f"[SemanticRAG] Loaded cached embeddings: "
                      f"{self.doc_embeddings.shape[0]} docs "
                      f"x {self.doc_embeddings.shape[1]} dims (instant)")
                return

        # 缓存无效或不存在 → 重新编码
        print(f"[SemanticRAG] Encoding {len(self.knowledge_entries)} documents "
              f"(this may take ~30s on first run)...")
        texts = [e['key_text'] for e in self.knowledge_entries]
        self.doc_embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        # L2 归一化
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self.doc_embeddings = self.doc_embeddings / norms

        # 写入缓存
        np.save(cache_path, self.doc_embeddings)
        with open(hash_path, 'w') as f:
            f.write(json_hash)
        print(f"[SemanticRAG] Built & cached index: {self.doc_embeddings.shape[0]} docs "
              f"x {self.doc_embeddings.shape[1]} dims")

    # ── 检索 ───────────────────────────────────────────────────────

    # 战术→中文 + 严重程度 + 攻击类别 映射
    TACTIC_CN = {
        'reconnaissance': '侦察', 'resource-development': '资源开发',
        'initial-access': '初始访问', 'execution': '执行',
        'persistence': '持久化', 'privilege-escalation': '权限提升',
        'defense-evasion': '防御规避', 'credential-access': '凭据访问',
        'discovery': '发现', 'lateral-movement': '横向移动',
        'collection': '数据收集', 'command-and-control': '命令与控制',
        'exfiltration': '数据窃取', 'impact': '影响',
    }
    SEVERITY_MAP = {
        'impact': 'critical', 'execution': 'high',
        'exfiltration': 'high', 'command-and-control': 'high',
        'credential-access': 'high', 'initial-access': 'high',
        'defense-evasion': 'high', 'persistence': 'medium',
        'privilege-escalation': 'medium', 'lateral-movement': 'medium',
        'collection': 'medium', 'discovery': 'low',
        'reconnaissance': 'low', 'resource-development': 'low',
    }

    def _summarize_en(self, text: str, max_len: int = 120) -> str:
        """截断英文描述为短摘要。"""
        if len(text) <= max_len:
            return text
        return text[:max_len].rsplit('.', 1)[0] + '.'

    def search(self, query_text: str, top_k: int = 3,
               min_similarity: float = 0.2) -> List[Dict]:
        """
        语义检索 MITRE 知识库。

        Returns:
            [{name, similarity, description, category, severity,
              mitre: {id, name, tactics, tactics_cn}, ...}, ...]
        """
        if not self._initialized:
            self.initialize()

        query_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        similarities = np.dot(query_emb, self.doc_embeddings.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < min_similarity:
                continue
            entry = self.knowledge_entries[idx]
            val = entry['value']
            tech_id = val.get('technique_id', entry.get('id', ''))
            tactics_en = val.get('tactics', [])
            tactics_cn = [self.TACTIC_CN.get(t, t) for t in tactics_en]
            primary_tactic = tactics_en[0] if tactics_en else ''

            results.append({
                # 前端兼容字段
                'name': val.get('technique_name', tech_id),
                'similarity': round(sim, 4),
                'description': self._summarize_en(val.get('description', '')),
                'category': primary_tactic,
                'severity': self.SEVERITY_MAP.get(primary_tactic, 'medium'),
                # MITRE 结构化信息
                'mitre': {
                    'id': tech_id,
                    'name': val.get('technique_name', ''),
                    'tactics': tactics_cn,
                    'tactics_en': tactics_en,
                    'platforms': val.get('platforms', []),
                    'detection': val.get('detection', ''),
                    'mitigation': val.get('mitigation', ''),
                },
            })
            if len(results) >= top_k:
                break

        return results

    def get_stats(self) -> Dict:
        return {
            'num_entries': len(self.knowledge_entries),
            'model_name': self.model_name,
            'embedding_dim': self.doc_embeddings.shape[1] if self.doc_embeddings is not None else None,
            'initialized': self._initialized,
        }
