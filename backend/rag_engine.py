"""
Spider-Sense v2 - RAG Engine
Lightweight vector similarity search over attack pattern knowledge base.
No external dependencies — uses numpy for vector ops.
"""
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class RAGEngine:
    """
    Lightweight RAG (Retrieval-Augmented Generation) engine.
    Stores attack pattern embeddings and retrieves similar patterns via cosine similarity.

    Knowledge base consists of:
    - Attack pattern vectors (from training data)
    - MITRE ATT&CK technique summaries
    - Historical detection records
    """

    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.attack_patterns = {}  # pattern_id -> embedding vector
        self.attack_metadata = {}  # pattern_id -> {name, mitre_id, description, severity}
        self.mitre_knowledge = {}  # mitre_id -> {name, tactics, description, platforms}
        self.history_embeddings = []  # list of (embedding, result_dict)
        self._loaded = False

    def load_knowledge_base(self):
        """Load all knowledge bases from disk."""
        # Load attack patterns
        patterns_path = os.path.join(self.knowledge_dir, 'attack_patterns.json')
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    pid = item['id']
                    self.attack_patterns[pid] = np.array(item['embedding'], dtype=np.float32)
                    self.attack_metadata[pid] = {
                        'name': item.get('name', ''),
                        'mitre_id': item.get('mitre_id', ''),
                        'description': item.get('description', ''),
                        'severity': item.get('severity', 'medium'),
                        'category': item.get('category', 'unknown'),
                    }

        # Load MITRE ATT&CK knowledge
        mitre_path = os.path.join(self.knowledge_dir, 'mitre_attack.json')
        if os.path.exists(mitre_path):
            with open(mitre_path, 'r', encoding='utf-8') as f:
                self.mitre_knowledge = json.load(f)

        # Initialize with defaults if empty
        if not self.attack_patterns:
            self._init_default_knowledge()

        self._loaded = True

    def _init_default_knowledge(self):
        """Initialize with built-in default knowledge base."""
        # Default attack pattern embeddings (simplified - real ones from training data)
        default_patterns = [
            {'id': 'ddos_flood', 'name': 'DDoS Flood Attack', 'mitre_id': 'T1498',
             'description': 'High-volume traffic flood targeting service availability. '
                            'Characterized by rapid packet rate, uniform packet sizes, '
                            'and traffic from distributed sources.',
             'severity': 'critical', 'category': 'ddos',
             'embedding': [0.8, 0.6, 0.9, 0.1, 0.3, 0.7, 0.2, 0.5] * 16},
            {'id': 'port_scan', 'name': 'Port Scanning', 'mitre_id': 'T1046',
             'description': 'Systematic probing of network ports to discover open services. '
                            'Characterized by sequential port access patterns and high '
                            'connection attempt rate to different ports.',
             'severity': 'medium', 'category': 'reconnaissance',
             'embedding': [0.2, 0.8, 0.1, 0.9, 0.4, 0.1, 0.7, 0.3] * 16},
            {'id': 'brute_force', 'name': 'Brute Force Attack', 'mitre_id': 'T1110',
             'description': 'Repeated authentication attempts to guess credentials. '
                            'Characterized by high frequency of login failures and '
                            'patterned credential submission.',
             'severity': 'high', 'category': 'credential_access',
             'embedding': [0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1] * 16},
            {'id': 'data_exfil', 'name': 'Data Exfiltration', 'mitre_id': 'T1041',
             'description': 'Unauthorized transfer of data from target network. '
                            'Characterized by unusual outbound traffic volume, '
                            'abnormal destination ports, and off-hours activity.',
             'severity': 'critical', 'category': 'exfiltration',
             'embedding': [0.1, 0.4, 0.2, 0.7, 0.5, 0.9, 0.3, 0.6] * 16},
            {'id': 'c2_beacon', 'name': 'C2 Beaconing', 'mitre_id': 'T1071',
             'description': 'Periodic communication with command-and-control server. '
                            'Characterized by regular interval connections, consistent '
                            'packet sizes, and encrypted payloads.',
             'severity': 'critical', 'category': 'command_and_control',
             'embedding': [0.7, 0.2, 0.5, 0.8, 0.1, 0.3, 0.9, 0.4] * 16},
            {'id': 'sql_injection', 'name': 'SQL Injection', 'mitre_id': 'T1190',
             'description': 'Injection of SQL commands through application inputs. '
                            'Characterized by SQL keywords in HTTP payloads and '
                            'unusual database query patterns.',
             'severity': 'high', 'category': 'web_attack',
             'embedding': [0.3, 0.9, 0.4, 0.6, 0.2, 0.5, 0.8, 0.1] * 16},
            {'id': 'xss', 'name': 'Cross-Site Scripting (XSS)', 'mitre_id': 'T1189',
             'description': 'Injection of malicious scripts into web pages. '
                            'Characterized by script tags in HTTP requests and '
                            'abnormal parameter encoding.',
             'severity': 'medium', 'category': 'web_attack',
             'embedding': [0.6, 0.7, 0.3, 0.5, 0.9, 0.2, 0.1, 0.4] * 16},
            {'id': 'malware_download', 'name': 'Malware Download', 'mitre_id': 'T1204',
             'description': 'Download of malicious executables from external sources. '
                            'Characterized by executable file signatures in traffic and '
                            'connections to known malicious IPs.',
             'severity': 'high', 'category': 'malware',
             'embedding': [0.4, 0.1, 0.8, 0.3, 0.6, 0.7, 0.5, 0.2] * 16},
        ]
        for p in default_patterns:
            self.attack_patterns[p['id']] = np.array(p['embedding'], dtype=np.float32)
            self.attack_metadata[p['id']] = {
                'name': p['name'], 'mitre_id': p['mitre_id'],
                'description': p['description'],
                'severity': p['severity'], 'category': p['category'],
            }

        # Default MITRE knowledge
        self.mitre_knowledge = {
            'T1498': {'name': 'Network Denial of Service', 'tactics': ['Impact'],
                      'description': 'Adversaries perform Network Denial of Service attacks.',
                      'platforms': ['Linux', 'Windows', 'macOS', 'Network']},
            'T1046': {'name': 'Network Service Scanning', 'tactics': ['Discovery'],
                      'description': 'Adversaries attempt to get a listing of services running on remote hosts.',
                      'platforms': ['Linux', 'Windows', 'macOS', 'Network']},
            'T1110': {'name': 'Brute Force', 'tactics': ['Credential Access'],
                      'description': 'Adversaries use brute force techniques to gain access to accounts.',
                      'platforms': ['Linux', 'Windows', 'macOS', 'Office 365', 'SaaS']},
            'T1041': {'name': 'Exfiltration Over C2 Channel', 'tactics': ['Exfiltration'],
                      'description': 'Adversaries steal data by exfiltrating it over a C2 channel.',
                      'platforms': ['Linux', 'Windows', 'macOS']},
            'T1071': {'name': 'Application Layer Protocol', 'tactics': ['Command and Control'],
                      'description': 'Adversaries communicate using application layer protocols.',
                      'platforms': ['Linux', 'Windows', 'macOS']},
            'T1190': {'name': 'Exploit Public-Facing Application', 'tactics': ['Initial Access'],
                      'description': 'Adversaries exploit vulnerabilities in public-facing applications.',
                      'platforms': ['Linux', 'Windows', 'macOS', 'Network']},
            'T1189': {'name': 'Drive-by Compromise', 'tactics': ['Initial Access'],
                      'description': 'Adversaries gain access through a user visiting a website.',
                      'platforms': ['Linux', 'Windows', 'macOS']},
            'T1204': {'name': 'User Execution', 'tactics': ['Execution'],
                      'description': 'Adversaries rely on user interaction for execution.',
                      'platforms': ['Linux', 'Windows', 'macOS']},
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def search(self, embedding: np.ndarray, top_k: int = 3,
               min_similarity: float = 0.1) -> List[Dict]:
        """
        Search for similar attack patterns given an embedding vector.

        Args:
            embedding: Feature embedding vector [D] (any dimension supported)
            top_k: Number of top results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of matched patterns with similarity scores and metadata
        """
        if not self._loaded:
            self.load_knowledge_base()

        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # Compute similarities to all stored patterns
        scores = []
        for pid, pattern_emb in self.attack_patterns.items():
            pattern_emb = np.asarray(pattern_emb, dtype=np.float32).flatten()
            # Adaptive dimension matching
            min_dim = min(len(embedding), len(pattern_emb))
            if min_dim < 4:
                continue
            # Use first N dims and last N dims for better coverage
            half = min_dim // 2
            a = np.concatenate([embedding[:half], embedding[-half:]])
            b = np.concatenate([pattern_emb[:half], pattern_emb[-half:]])
            sim = self._cosine_similarity(a, b)
            if sim >= min_similarity:
                scores.append((pid, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-K
        results = []
        for pid, sim in scores[:top_k]:
            meta = self.attack_metadata.get(pid, {})
            mitre_id = meta.get('mitre_id', '')
            mitre_info = self.mitre_knowledge.get(mitre_id, {})

            results.append({
                'pattern_id': pid,
                'similarity': round(sim, 4),
                'name': meta.get('name', pid),
                'description': meta.get('description', ''),
                'severity': meta.get('severity', 'unknown'),
                'category': meta.get('category', 'unknown'),
                'mitre': {
                    'id': mitre_id,
                    'name': mitre_info.get('name', ''),
                    'tactics': mitre_info.get('tactics', []),
                    'description': mitre_info.get('description', ''),
                } if mitre_id else None,
            })

        return results

    def add_to_history(self, embedding: np.ndarray, result: Dict):
        """Record a detection result into the knowledge base."""
        self.history_embeddings.append((np.asarray(embedding), result))
        # Keep only recent 10000 entries
        if len(self.history_embeddings) > 10000:
            self.history_embeddings = self.history_embeddings[-10000:]

    def search_history(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search historical detections for similar cases."""
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        scores = []
        for i, (hist_emb, result) in enumerate(self.history_embeddings):
            min_dim = min(len(embedding), len(hist_emb))
            sim = self._cosine_similarity(embedding[:min_dim], hist_emb[:min_dim])
            scores.append((i, sim, result))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'similarity': round(s, 4), 'result': r} for _, s, r in scores[:top_k]]

    def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics."""
        if not self._loaded:
            self.load_knowledge_base()
        return {
            'num_patterns': len(self.attack_patterns),
            'num_mitre_techniques': len(self.mitre_knowledge),
            'num_history_entries': len(self.history_embeddings),
            'categories': list(set(m.get('category', '') for m in self.attack_metadata.values())),
            'loaded': self._loaded,
        }
