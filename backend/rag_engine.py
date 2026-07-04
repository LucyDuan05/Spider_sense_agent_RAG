"""
Spider-Sense v2 - Knowledge Manager
MITRE ATT&CK 知识库加载 + 检测历史追踪。

语义 RAG 检索已迁移至 semantic_rag.py。
本模块保留: MITRE 知识加载、历史检测记录、知识库统计。
"""
import json
import os
import numpy as np
from typing import Dict, List


class RAGEngine:
    """
    Knowledge manager — 加载 MITRE ATT&CK JSON，追踪检测历史。

    注意: 攻击模式的向量检索已由 semantic_rag.SemanticRAG 接管。
    本类不再维护 attack_patterns 潜向量索引。
    """

    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.mitre_knowledge = {}       # mitre_id → {name, tactics, description, ...}
        self.history_embeddings = []    # list of (embedding, result_dict)
        self._loaded = False

    # ── MITRE 知识库加载 ───────────────────────────────────────────

    def load_knowledge_base(self):
        """加载 MITRE ATT&CK 知识库 JSON。

        优先加载 mitre_semantic.json (858 条, 从 enterprise-attack.json 生成),
        回退到 mitre_attack.json (29 条, 手写).
        """
        # 优先: 完整版语义知识库
        semantic_path = os.path.join(self.knowledge_dir, 'mitre_semantic.json')
        mitre_path = os.path.join(self.knowledge_dir, 'mitre_attack.json')

        if os.path.exists(semantic_path):
            self._load_semantic_mitre(semantic_path)
        elif os.path.exists(mitre_path):
            self._load_legacy_mitre(mitre_path)

        if not self.mitre_knowledge:
            self._init_fallback_mitre()

        self._loaded = True
        print(f"[Knowledge] Loaded {len(self.mitre_knowledge)} MITRE ATT&CK techniques")

    def _load_semantic_mitre(self, path: str):
        """加载 mitre_semantic.json (新格式: [{id, key_text, value}, ...])."""
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        for entry in entries:
            val = entry.get('value', {})
            tech_id = val.get('technique_id', entry.get('id', ''))
            if not tech_id:
                continue
            self.mitre_knowledge[tech_id] = {
                'name': val.get('technique_name', val.get('technique_id', '')),
                'tactics': val.get('tactics', []),
                'tactics_cn': val.get('tactics_cn', ''),
                'description': val.get('description', ''),
                'detection': val.get('detection', ''),
                'platforms': val.get('platforms', []),
                'data_sources': val.get('data_sources', []),
                'source': 'mitre_semantic',
            }
        print(f"[Knowledge] Loaded semantic MITRE: {len(entries)} entries → {len(self.mitre_knowledge)} techniques")

    def _load_legacy_mitre(self, path: str):
        """加载旧格式 mitre_attack.json ({tech_id: {...}})."""
        with open(path, 'r', encoding='utf-8') as f:
            self.mitre_knowledge = json.load(f)
        print(f"[Knowledge] Loaded legacy MITRE: {len(self.mitre_knowledge)} techniques")

    def _init_fallback_mitre(self):
        """Fallback MITRE 知识 (打包环境 JSON 缺失时使用)。"""
        self.mitre_knowledge = {
            'T1498': {
                'name': 'Network Denial of Service',
                'tactics': ['Impact'],
                'description': '攻击者通过耗尽网络带宽或资源来执行网络拒绝服务攻击。',
                'platforms': ['Linux', 'Windows', 'macOS', 'Network'],
                'detection': '监控网络流量基线偏差、入站流量速率异常',
                'mitigation': '流量清洗、速率限制、CDN分发、上游ISP过滤',
            },
            'T1046': {
                'name': 'Network Service Scanning',
                'tactics': ['Discovery'],
                'description': '攻击者通过扫描远程主机获取运行中的服务列表。',
                'platforms': ['Linux', 'Windows', 'macOS', 'Network'],
                'detection': '监控短时间内多端口连接尝试',
                'mitigation': '网络分段、防火墙规则、端口敲门技术',
            },
            'T1110': {
                'name': 'Brute Force',
                'tactics': ['Credential Access'],
                'description': '攻击者通过暴力破解技术获取账户访问权限。',
                'platforms': ['Linux', 'Windows', 'macOS'],
                'detection': '监控登录失败率、账户锁定事件',
                'mitigation': '账户锁定策略、多因素认证、强密码策略',
            },
            'T1573': {
                'name': 'Encrypted Channel',
                'tactics': ['Command and Control'],
                'description': '攻击者使用加密通道隐藏C2流量。',
                'platforms': ['Linux', 'Windows', 'macOS'],
                'detection': 'TLS证书分析、JA3指纹识别',
                'mitigation': 'SSL/TLS解密代理、证书固定',
            },
        }

    def get_mitre_technique(self, technique_id: str) -> Dict:
        """获取单个 MITRE 技术的完整信息。"""
        if not self._loaded:
            self.load_knowledge_base()
        return self.mitre_knowledge.get(technique_id, {})

    # ── 检测历史追踪 ──────────────────────────────────────────────

    def add_to_history(self, embedding: np.ndarray, result: Dict):
        """记录一次检测结果到历史库。"""
        self.history_embeddings.append((np.asarray(embedding), result))
        if len(self.history_embeddings) > 10000:
            self.history_embeddings = self.history_embeddings[-10000:]

    def search_history(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """搜索历史检测中的相似案例 (特征空间 cos_sim)。"""
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        scores = []
        for i, (hist_emb, result) in enumerate(self.history_embeddings):
            min_dim = min(len(embedding), len(hist_emb))
            if min_dim < 4:
                continue
            a, b = embedding[:min_dim], hist_emb[:min_dim]
            a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
            sim = float(np.dot(a, b) / (a_norm * b_norm + 1e-10))
            scores.append((i, sim, result))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'similarity': round(s, 4), 'result': r}
                for _, s, r in scores[:top_k]]

    # ── 统计 ──────────────────────────────────────────────────────

    def get_knowledge_stats(self) -> Dict:
        """知识库统计 (供 /api/health 和 /api/knowledge/stats)。"""
        if not self._loaded:
            self.load_knowledge_base()
        return {
            'num_mitre_techniques': len(self.mitre_knowledge),
            'num_history_entries': len(self.history_embeddings),
            'loaded': self._loaded,
        }
