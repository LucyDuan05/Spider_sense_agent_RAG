"""
Spider-Sense v2 - Flow-to-Text Converter
78-dim CICIDS features → natural language behavioral description.

Two output modes:
  to_rag_query()  → English, keyword-dense  → semantic embedding → MITRE search
  to_agent_context() → Chinese, structured → LLM prompt / frontend display

Design principle:
  The converter doesn't classify attacks — it describes observable behaviors.
  MITRE ATT&CK entries are also written as observable behaviors.
  When both describe the same phenomena, their embeddings align in vector space.
"""
import numpy as np
from typing import Dict, Optional


# ── Port groups with expected behavioral baseline ──────────────────
# Each port group maps to a typical service and its normal traffic profile.
PORT_GROUPS = {
    21:  ('FTP',      'file_transfer',   'bulk data transfer, authentication'),
    22:  ('SSH',      'remote_admin',    'interactive session, encrypted'),
    23:  ('Telnet',   'remote_admin',    'interactive session, plaintext'),
    25:  ('SMTP',     'mail',            'email delivery'),
    53:  ('DNS',      'name_resolution', 'small UDP queries, occasional TCP zone transfer'),
    80:  ('HTTP',     'web',             'request-response, variable payload sizes'),
    110: ('POP3',     'mail',            'email retrieval'),
    143: ('IMAP',     'mail',            'email sync'),
    443: ('HTTPS',    'web_encrypted',   'encrypted request-response, uniform packet pacing'),
    445: ('SMB',      'file_share',      'LAN file operations, often high-volume'),
    1433: ('MSSQL',   'database',        'structured queries, result sets'),
    3306: ('MySQL',   'database',        'structured queries, result sets'),
    3389: ('RDP',     'remote_desktop',  'interactive GUI session, encrypted'),
    5432: ('PostgreSQL', 'database',     'structured queries, result sets'),
    6379: ('Redis',   'cache',           'key-value ops, low latency'),
    8080: ('HTTP-Alt','web',             'alternative HTTP, often dev/internal'),
    27017: ('MongoDB','database',        'document queries'),
}


# ── Security-relevant thresholds (empirical, network forensics domain) ──

class Thresholds:
    """Empirical thresholds grounded in network security observation."""

    # Packet rate (pkts/s)
    PKT_RATE_EXTREME = 10000    # clearly DDoS flood territory
    PKT_RATE_HIGH    = 1000     # suspiciously high
    PKT_RATE_LOW     = 10       # below this, not bulk data movement

    # Byte rate (bytes/s)
    BYTE_RATE_EXTREME = 10_000_000  # 10 MB/s — volumetric attack
    BYTE_RATE_HIGH    =  1_000_000  # 1 MB/s
    BYTE_RATE_LOW     =      1_000  # 1 KB/s — anomalous for data transfer

    # Packet size uniformity (standard deviation in bytes)
    PKT_UNIFORM_STD = 30     # std < 30 → tool-generated, automated
    PKT_VARIABLE_STD = 200   # std > 200 → human interactive, mixed content

    # Duration (seconds; raw feature is microseconds)
    DURATION_SCAN = 0.1       # < 100ms → probe/scan
    DURATION_SHORT = 1.0      # < 1s
    DURATION_LONG = 60.0      # > 1 min
    DURATION_VERY_LONG = 3600 # > 1 hour

    # TCP flag ratios
    SYN_FLOOD_RATIO = 0.80    # >80% SYN → SYN flood
    RST_SCAN_RATIO = 0.30     # >30% RST → port scan
    ACK_FLOOD_RATIO = 0.90    # >90% ACK → ACK flood

    # Direction asymmetry
    INBOUND_HEAVY = 100       # down/up > 100 → mostly inbound (DDoS, scan responses)
    OUTBOUND_HEAVY = 0.01     # down/up < 0.01 → mostly outbound (data exfil)

    # Slow attack indicators
    SLOW_CONN_PKTS = 50       # many packets
    SLOW_RATE = 10            # but low rate → slow connection exhaustion

    # CROSR reconstruction
    RECON_ANOMALY = 0.15      # above → decoder can't reconstruct → out-of-distribution


class FlowToText:
    """
    Converts a 78-dim CICIDS feature vector into a behavioral description.

    Usage:
        converter = FlowToText(scaler_path='processed_cicids/scaler.npz')
        eng_text = converter.to_rag_query(features, detection)
        chn_text = converter.to_agent_context(features, detection)
    """

    def __init__(self, scaler_path: str = None):
        self.feature_names = []
        self._name_to_idx = {}
        self._mean = None
        self._scale = None

        if scaler_path:
            data = np.load(scaler_path, allow_pickle=True)
            self.feature_names = [str(n) for n in data['feature_names'].tolist()]
            self._mean = data['mean'].astype(np.float32)
            self._scale = data['scale'].astype(np.float32)
            self._scale = np.where(self._scale < 1e-10, 1.0, self._scale)
            self._name_to_idx = {n: i for i, n in enumerate(self.feature_names)}

    # ── Internal helpers ──────────────────────────────────────────

    def _raw(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=np.float32).flatten()
        if self._mean is not None:
            return f * self._scale + self._mean
        return f

    def _v(self, raw: np.ndarray, name: str, default: float = 0.0) -> float:
        idx = self._name_to_idx.get(name, -1)
        if idx >= 0 and idx < len(raw):
            return float(raw[idx])
        return default

    def _port_info(self, raw: np.ndarray) -> tuple:
        """Returns (port_number, service_name, service_category)."""
        port = int(self._v(raw, 'Destination Port'))
        name, category, _ = PORT_GROUPS.get(port, (str(port), 'unknown', ''))
        return port, name, category

    # ── Public API ─────────────────────────────────────────────────

    def to_rag_query(self, features: np.ndarray,
                     detection: Optional[Dict] = None) -> str:
        """
        English behavioral description for semantic embedding and MITRE search.

        Output is keyword-dense, fact-based, using observable-behavior language
        that matches the style of MITRE ATT&CK key_text entries.
        """
        raw = self._raw(features)
        parts = []

        # ── What is being targeted ──
        port, service, category = self._port_info(raw)
        parts.append(f"Target: port {port} ({service}), service category: {category}.")

        # ── Traffic volume ──
        pkts_s = self._v(raw, 'Flow Packets/s')
        bytes_s = self._v(raw, 'Flow Bytes/s')
        fwd_pkts = int(self._v(raw, 'Total Fwd Packets'))
        bwd_pkts = int(self._v(raw, 'Total Backward Packets'))
        fwd_bytes = self._v(raw, 'Total Length of Fwd Packets')
        bwd_bytes = self._v(raw, 'Total Length of Bwd Packets')

        if pkts_s > Thresholds.PKT_RATE_EXTREME:
            parts.append(f"Extreme packet rate: {pkts_s:.0f} pkts/s, "
                         f"indicative of volumetric flood attack.")
        elif pkts_s > Thresholds.PKT_RATE_HIGH:
            parts.append(f"High packet rate: {pkts_s:.0f} pkts/s.")
        elif pkts_s < Thresholds.PKT_RATE_LOW:
            parts.append(f"Low packet rate: {pkts_s:.2f} pkts/s, "
                         f"not consistent with bulk data transfer.")

        if bytes_s > Thresholds.BYTE_RATE_EXTREME:
            parts.append(f"Extreme bandwidth: {bytes_s/1e6:.1f} MB/s, "
                         f"saturating network capacity.")
        elif bytes_s > Thresholds.BYTE_RATE_HIGH:
            parts.append(f"High bandwidth: {bytes_s/1e3:.0f} KB/s.")
        elif bytes_s < Thresholds.BYTE_RATE_LOW:
            parts.append(f"Very low throughput: {bytes_s:.1f} B/s, "
                         f"suggesting idle connection or stealth activity.")

        parts.append(f"Forward: {fwd_pkts} packets ({fwd_bytes:.0f} bytes). "
                     f"Backward: {bwd_pkts} packets ({bwd_bytes:.0f} bytes).")

        # ── Packet size distribution ──
        pkt_mean = self._v(raw, 'Packet Length Mean')
        pkt_std = self._v(raw, 'Packet Length Std')

        if pkt_std < Thresholds.PKT_UNIFORM_STD and pkt_mean > 50:
            parts.append(f"Uniform packet sizes: mean {pkt_mean:.0f}B, std {pkt_std:.0f}B. "
                         f"Consistent with automated attack tool output "
                         f"(DDoS bots, brute-force scripts, scanning tools).")
        elif pkt_std > Thresholds.PKT_VARIABLE_STD:
            parts.append(f"Highly variable packet sizes: mean {pkt_mean:.0f}B, "
                         f"std {pkt_std:.0f}B. Typical of human-driven interactive traffic.")
        elif pkt_mean < 10:
            parts.append(f"Tiny packets: mean {pkt_mean:.0f}B. "
                         f"Consistent with scanning probes, ACK/SYN handshakes, "
                         f"or command-and-control beacons.")

        # ── TCP flags → attack signature ──
        total_pkts = fwd_pkts + bwd_pkts
        syn = self._v(raw, 'SYN Flag Count')
        ack = self._v(raw, 'ACK Flag Count')
        rst = self._v(raw, 'RST Flag Count')
        fin = self._v(raw, 'FIN Flag Count')
        psh = self._v(raw, 'PSH Flag Count')

        if total_pkts > 0:
            syn_r = syn / total_pkts
            rst_r = rst / total_pkts
            ack_r = ack / total_pkts
            if syn_r > Thresholds.SYN_FLOOD_RATIO:
                parts.append(f"SYN flood signature: {syn_r:.0%} of packets are SYN, "
                             f"indicating half-open connection flood (TCP SYN flood DDoS).")
            if rst_r > Thresholds.RST_SCAN_RATIO:
                parts.append(f"High RST ratio: {rst_r:.0%}. "
                             f"Typical of port scanning where most probes are rejected.")
            if ack_r > Thresholds.ACK_FLOOD_RATIO:
                parts.append(f"ACK flood signature: {ack_r:.0%} ACK packets. "
                             f"Possible ACK flood DDoS attempting to bypass stateless filters.")
            if fin > 0 and total_pkts < 10:
                parts.append("Clean connection teardown with FIN handshake.")
            if psh > 0 and total_pkts < 5:
                parts.append("Push flag set, small data push (PSH).")

        # ── Connection patterns ──
        dur_us = self._v(raw, 'Flow Duration')
        dur_s = dur_us / 1_000_000.0
        down_up = self._v(raw, 'Down/Up Ratio')

        if dur_s < Thresholds.DURATION_SCAN:
            parts.append(f"Extremely short duration ({dur_s*1000:.1f}ms). "
                         f"Consistent with port scanning, DNS query, or failed handshake.")
        elif dur_s > Thresholds.DURATION_VERY_LONG:
            parts.append(f"Extremely long duration ({dur_s/3600:.1f} hours). "
                         f"Possible persistent C2 channel or data exfiltration session.")
        elif dur_s > Thresholds.DURATION_LONG:
            parts.append(f"Long duration ({dur_s:.0f}s).")

        if down_up > Thresholds.INBOUND_HEAVY:
            parts.append(f"Highly asymmetric traffic: {down_up:.0f}x more inbound than "
                         f"outbound. Consistent with DDoS attack or scan results.")
        elif down_up < Thresholds.OUTBOUND_HEAVY and (fwd_bytes > 0 or bwd_bytes > 0):
            parts.append("Traffic predominantly outbound. Possible data exfiltration "
                         "or command-and-control beaconing.")

        # Slow attack pattern
        if total_pkts > Thresholds.SLOW_CONN_PKTS and pkts_s < Thresholds.SLOW_RATE:
            parts.append(f"Slow connection exhaustion pattern: {total_pkts:.0f} connections "
                         f"at only {pkts_s:.2f} pkts/s. Characteristic of Slowloris, "
                         f"Slow HTTP POST, or application-layer resource depletion.")

        # ── Detection context ──
        if detection:
            if detection.get('is_unknown', False):
                us = detection.get('unknown_score', 0)
                parts.append(f"OpenMax verdict: UNKNOWN (open-set rejection, "
                             f"unknown probability {us:.0%}). "
                             f"This flow does not match any of the 6 trained classes.")
            else:
                parts.append(f"Model classification: {detection.get('prediction', 'N/A')}.")

            if detection.get('recon_error', 0) > Thresholds.RECON_ANOMALY:
                parts.append(f"High CROSR reconstruction error ({detection['recon_error']:.4f}), "
                             f"confirming out-of-distribution pattern.")

        return ' '.join(parts)

    def to_agent_context(self, features: np.ndarray,
                         detection: Optional[Dict] = None) -> str:
        """
        Chinese structured description for LLM Agent prompt and frontend display.
        Organized as a mini threat briefing: target → behavior → verdict.
        """
        raw = self._raw(features)
        lines = []

        # ── 目标信息 ──
        port, service, category = self._port_info(raw)
        cat_cn = {
            'file_transfer': '文件传输', 'remote_admin': '远程管理',
            'mail': '邮件服务', 'name_resolution': '域名解析',
            'web': 'Web服务', 'web_encrypted': '加密Web服务',
            'file_share': '文件共享', 'database': '数据库',
            'remote_desktop': '远程桌面', 'cache': '缓存服务',
            'unknown': '未知服务',
        }.get(category, category)
        lines.append(f"【目标】端口 {port} ({service})，服务类型: {cat_cn}。")

        # ── 流量体量 ──
        pkts_s = self._v(raw, 'Flow Packets/s')
        bytes_s = self._v(raw, 'Flow Bytes/s')
        fwd_pkts = int(self._v(raw, 'Total Fwd Packets'))
        bwd_pkts = int(self._v(raw, 'Total Backward Packets'))
        fwd_bytes = self._v(raw, 'Total Length of Fwd Packets')
        bwd_bytes = self._v(raw, 'Total Length of Bwd Packets')

        if pkts_s > Thresholds.PKT_RATE_EXTREME:
            lines.append(f"包速率: {pkts_s:.0f} pkt/s（极端高频，符合洪泛攻击特征）。")
        elif pkts_s > Thresholds.PKT_RATE_HIGH:
            lines.append(f"包速率: {pkts_s:.0f} pkt/s（较高）。")
        elif pkts_s < Thresholds.PKT_RATE_LOW:
            lines.append(f"包速率: {pkts_s:.2f} pkt/s（极低，非批量传输模式）。")

        if bytes_s > Thresholds.BYTE_RATE_EXTREME:
            lines.append(f"带宽: {bytes_s/1e6:.1f} MB/s（极端带宽消耗，网络容量饱和）。")
        elif bytes_s > Thresholds.BYTE_RATE_HIGH:
            lines.append(f"带宽: {bytes_s/1e3:.0f} KB/s（较高）。")
        elif bytes_s < Thresholds.BYTE_RATE_LOW:
            lines.append(f"带宽: {bytes_s:.1f} B/s（极低吞吐量，可能为空闲连接或隐蔽活动）。")

        lines.append(f"前向: {fwd_pkts} 包 ({fwd_bytes:.0f} B)，"
                     f"后向: {bwd_pkts} 包 ({bwd_bytes:.0f} B)。")

        # ── 包大小分布 ──
        pkt_mean = self._v(raw, 'Packet Length Mean')
        pkt_std = self._v(raw, 'Packet Length Std')

        if pkt_std < Thresholds.PKT_UNIFORM_STD and pkt_mean > 50:
            lines.append(f"包大小分布: 高度均匀（均值 {pkt_mean:.0f} B，标准差 {pkt_std:.0f} B）。"
                         f"这说明流量由自动化工具生成（如 DDoS 脚本、暴力破解工具、扫描器），"
                         f"而非人工交互。")
        elif pkt_std > Thresholds.PKT_VARIABLE_STD:
            lines.append(f"包大小分布: 高度变异（均值 {pkt_mean:.0f} B，标准差 {pkt_std:.0f} B）。"
                         f"这符合人工交互流量的特征。")
        elif pkt_mean < 10:
            lines.append(f"包大小: 极小（均值 {pkt_mean:.0f} B）。"
                         f"典型于扫描探测、握手包或 C2 信标。")

        # ── TCP 标志 ──
        total_pkts = fwd_pkts + bwd_pkts
        syn = self._v(raw, 'SYN Flag Count')
        ack = self._v(raw, 'ACK Flag Count')
        rst = self._v(raw, 'RST Flag Count')

        if total_pkts > 0:
            syn_r = syn / total_pkts
            rst_r = rst / total_pkts
            ack_r = ack / total_pkts
            if syn_r > Thresholds.SYN_FLOOD_RATIO:
                lines.append(f"SYN 洪泛特征: {syn_r:.0%} 的包为 SYN，"
                             f"表明大量半开连接（TCP SYN Flood DDoS）。")
            if rst_r > Thresholds.RST_SCAN_RATIO:
                lines.append(f"RST 比例偏高: {rst_r:.0%}。"
                             f"大量连接被目标拒绝，常见的端口扫描特征。")
            if ack_r > Thresholds.ACK_FLOOD_RATIO:
                lines.append(f"ACK 洪泛特征: {ack_r:.0%} 为 ACK 包，"
                             f"可能试图绕过无状态防火墙。")

        # ── 连接模式 ──
        dur_s = self._v(raw, 'Flow Duration') / 1_000_000.0
        down_up = self._v(raw, 'Down/Up Ratio')

        if dur_s < Thresholds.DURATION_SCAN:
            lines.append(f"连接持续时间: 极短（{dur_s*1000:.1f} ms），"
                         f"符合端口扫描、DNS 查询或握手失败的特征。")
        elif dur_s > Thresholds.DURATION_VERY_LONG:
            lines.append(f"连接持续时间: 极长（{dur_s/3600:.1f} 小时），"
                         f"可能为持久化 C2 通道或数据外传会话。")

        if down_up > Thresholds.INBOUND_HEAVY:
            lines.append(f"流量方向: 严重不对称（下行/上行 = {down_up:.0f}），"
                         f"主要为入站流量，符合 DDoS 攻击或扫描响应的特征。")
        elif down_up < Thresholds.OUTBOUND_HEAVY and (fwd_bytes > 0 or bwd_bytes > 0):
            lines.append("流量方向: 主要为出站方向，可能存在数据外传或 C2 信标行为。")

        if total_pkts > Thresholds.SLOW_CONN_PKTS and pkts_s < Thresholds.SLOW_RATE:
            lines.append(f"慢速攻击模式: {total_pkts:.0f} 个连接但速率仅 {pkts_s:.2f} pkt/s，"
                         f"这是 Slowloris / Slow HTTP POST 等应用层连接耗尽攻击的典型特征。")

        # ── 检测结果 ──
        if detection:
            pred = detection.get('prediction', 'N/A')
            is_unknown = detection.get('is_unknown', False)
            us = detection.get('unknown_score', 0)
            recon = detection.get('recon_error', 0)

            if is_unknown:
                lines.append(f"【OpenMax 判定】UNKNOWN（未知攻击，未知概率 {us:.0%}）。"
                             f"该流量模式与 6 种已知攻击类别均不匹配。")
            else:
                lines.append(f"【模型判定】{pred}。")

            if recon > Thresholds.RECON_ANOMALY:
                lines.append(f"CROSR 重建误差 {recon:.4f}（超过阈值），"
                             f"确认该流量处于训练分布之外。")

        return '\n'.join(lines)
