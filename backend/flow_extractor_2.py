"""
Spider-Sense v2 - Flow-based Feature Extractor v2
Reassembles packets into flows and computes CICIDS-2017 style 78-dim features.
Uses training scaler (mean/scale) for proper normalization.
"""
import os, time, numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from scapy.all import IP, TCP, UDP, Raw
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

# ── Constants ────────────────────────────────────────────────────────
FLOW_TIMEOUT = 120.0
MAX_FLOWS = 10000
TCP_PROTO, UDP_PROTO = 6, 17
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'processed_cicids', 'scaler.npz')


class FeatureNormalizer:
    """Loads training scaler; normalizes raw features via (x - mean) / scale."""

    def __init__(self, scaler_path: str = SCALER_PATH):
        data = np.load(scaler_path, allow_pickle=True)
        self.mean = data['mean'].astype(np.float32)
        self.scale = data['scale'].astype(np.float32)
        self.feature_names = [str(n) for n in data['feature_names'].tolist()]
        self.dim = len(self.feature_names)
        # Safety: replace zero scale to avoid division by zero
        self.scale = np.where(self.scale < 1e-10, 1.0, self.scale)

    def normalize(self, raw_vec: np.ndarray) -> np.ndarray:
        """Apply StandardScaler normalization."""
        return ((raw_vec - self.mean) / self.scale).astype(np.float32)


# ── Flow State ───────────────────────────────────────────────────────

class FlowState:
    """Tracks state of a single bi-directional network flow for CICIDS feature extraction."""

    __slots__ = (
        'key', 'protocol', 'first_seen', 'last_seen', 'last_pkt_time',
        'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes',
        'fwd_pkt_lens', 'bwd_pkt_lens', 'fwd_iat', 'bwd_iat',
        'fwd_init_win', 'bwd_init_win',
        'syn_cnt', 'ack_cnt', 'fin_cnt', 'rst_cnt', 'psh_cnt', 'urg_cnt',
        'act_data_pkt_fwd', 'active_start', 'total_active', 'total_idle',
    )

    def __init__(self, key: tuple, protocol: int, first_pkt_time: float):
        self.key = key
        self.protocol = protocol
        self.first_seen = first_pkt_time
        self.last_seen = first_pkt_time
        self.last_pkt_time = first_pkt_time
        self.fwd_pkts = self.bwd_pkts = 0
        self.fwd_bytes = self.bwd_bytes = 0
        self.fwd_pkt_lens: List[float] = []
        self.bwd_pkt_lens: List[float] = []
        self.fwd_iat: List[float] = []
        self.bwd_iat: List[float] = []
        self.fwd_init_win = self.bwd_init_win = 0
        self.syn_cnt = self.ack_cnt = self.fin_cnt = 0
        self.rst_cnt = self.psh_cnt = self.urg_cnt = 0
        self.act_data_pkt_fwd = 0
        self.total_active = 0.0
        self.total_idle = 0.0

    def update(self, pkt, direction: str, pkt_time: float):
        """Update flow state with a new packet."""
        self.last_seen = pkt_time
        pkt_len = len(pkt)
        iat = pkt_time - self.last_pkt_time if self.last_pkt_time > 0 else 0

        if direction == 'fwd':
            self.fwd_pkts += 1
            self.fwd_bytes += pkt_len
            self.fwd_pkt_lens.append(pkt_len)
            if self.fwd_pkts > 1:
                self.fwd_iat.append(iat)
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += pkt_len
            self.bwd_pkt_lens.append(pkt_len)
            if self.bwd_pkts > 1:
                self.bwd_iat.append(iat)
        if iat > 1.0:
            self.total_idle += iat
        else:
            self.total_active += iat

        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags.S: self.syn_cnt += 1
            if flags.A: self.ack_cnt += 1
            if flags.F: self.fin_cnt += 1
            if flags.R: self.rst_cnt += 1
            if flags.P: self.psh_cnt += 1
            if flags.U: self.urg_cnt += 1
            if direction == 'fwd' and self.fwd_pkts == 1:
                self.fwd_init_win = pkt[TCP].window
            if direction == 'bwd' and self.bwd_pkts == 1:
                self.bwd_init_win = pkt[TCP].window
            if flags.A and pkt_len > 0:
                self.act_data_pkt_fwd += 1

        self.last_pkt_time = pkt_time

    def is_complete(self, current_time: float) -> bool:
        if self.fin_cnt > 0 or self.rst_cnt > 0:
            return True
        if current_time - self.last_seen > FLOW_TIMEOUT:
            return True
        return False

    def build_raw_vector(self) -> np.ndarray:
        """
        Build raw 78-dim feature vector using CICIDS-2017 feature names,
        matching the order from the training scaler.
        """
        total = self.fwd_pkts + self.bwd_pkts
        duration = max(self.last_seen - self.first_seen, 0.0001)
        if self.fwd_pkt_lens:
            fwd_len_mean = float(np.mean(self.fwd_pkt_lens))
            fwd_len_std = float(np.std(self.fwd_pkt_lens)) if len(self.fwd_pkt_lens) > 1 else 0
        else:
            fwd_len_mean = fwd_len_std = 0
        if self.bwd_pkt_lens:
            bwd_len_mean = float(np.mean(self.bwd_pkt_lens))
        else:
            bwd_len_mean = 0
        all_lens = self.fwd_pkt_lens + self.bwd_pkt_lens
        if all_lens:
            pkt_len_mean = float(np.mean(all_lens))
            pkt_len_std = float(np.std(all_lens)) if len(all_lens) > 1 else 0
            pkt_len_var = float(np.var(all_lens)) if len(all_lens) > 1 else 0
            pkt_len_min = float(np.min(all_lens))
            pkt_len_max = float(np.max(all_lens))
        else:
            pkt_len_mean = pkt_len_std = pkt_len_var = 0
            pkt_len_min = pkt_len_max = 0

        if self.fwd_iat:
            fwd_iat_mean = float(np.mean(self.fwd_iat))
            fwd_iat_std = float(np.std(self.fwd_iat)) if len(self.fwd_iat) > 1 else 0
            fwd_iat_max = float(np.max(self.fwd_iat))
            fwd_iat_min = float(np.min(self.fwd_iat))
            fwd_iat_total = float(np.sum(self.fwd_iat))
        else:
            fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = fwd_iat_total = 0
        if self.bwd_iat:
            bwd_iat_mean = float(np.mean(self.bwd_iat))
            bwd_iat_std = float(np.std(self.bwd_iat)) if len(self.bwd_iat) > 1 else 0
            bwd_iat_max = float(np.max(self.bwd_iat))
            bwd_iat_min = float(np.min(self.bwd_iat))
            bwd_iat_total = float(np.sum(self.bwd_iat))
        else:
            bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = bwd_iat_total = 0

        # Order-sensitive: this dict must match scaler['feature_names'] exactly
        raw = {
            'Destination Port': 0.0,
            'Flow Duration': duration * 1e6,  # seconds → microseconds
            'Total Fwd Packets': float(self.fwd_pkts),
            'Total Backward Packets': float(self.bwd_pkts),
            'Total Length of Fwd Packets': float(self.fwd_bytes),
            'Total Length of Bwd Packets': float(self.bwd_bytes),
            'Fwd Packet Length Max': float(max(self.fwd_pkt_lens)) if self.fwd_pkt_lens else 0,
            'Fwd Packet Length Min': float(min(self.fwd_pkt_lens)) if self.fwd_pkt_lens else 0,
            'Fwd Packet Length Mean': fwd_len_mean,
            'Fwd Packet Length Std': fwd_len_std,
            'Bwd Packet Length Max': float(max(self.bwd_pkt_lens)) if self.bwd_pkt_lens else 0,
            'Bwd Packet Length Min': float(min(self.bwd_pkt_lens)) if self.bwd_pkt_lens else 0,
            'Bwd Packet Length Mean': bwd_len_mean,
            'Bwd Packet Length Std': float(np.std(self.bwd_pkt_lens)) if len(self.bwd_pkt_lens) > 1 else 0,
            'Flow Bytes/s': (self.fwd_bytes + self.bwd_bytes) / duration,
            'Flow Packets/s': total / duration,
            'Flow IAT Mean': fwd_iat_mean if self.fwd_iat else 0,
            'Flow IAT Std': fwd_iat_std if self.fwd_iat else 0,
            'Flow IAT Max': fwd_iat_max if self.fwd_iat else 0,
            'Flow IAT Min': fwd_iat_min if self.fwd_iat else 0,
            'Fwd IAT Total': fwd_iat_total,
            'Fwd IAT Mean': fwd_iat_mean,
            'Fwd IAT Std': fwd_iat_std,
            'Fwd IAT Max': fwd_iat_max,
            'Fwd IAT Min': fwd_iat_min,
            'Bwd IAT Total': bwd_iat_total,
            'Bwd IAT Mean': bwd_iat_mean,
            'Bwd IAT Std': bwd_iat_std,
            'Bwd IAT Max': bwd_iat_max,
            'Bwd IAT Min': bwd_iat_min,
            'Fwd PSH Flags': float(self.psh_cnt),
            'Bwd PSH Flags': 0.0,
            'Fwd URG Flags': float(self.urg_cnt),
            'Bwd URG Flags': 0.0,
            'Fwd Header Length': 0.0,
            'Bwd Header Length': 0.0,
            'Fwd Packets/s': self.fwd_pkts / duration,
            'Bwd Packets/s': self.bwd_pkts / duration,
            'Min Packet Length': pkt_len_min,
            'Max Packet Length': pkt_len_max,
            'Packet Length Mean': pkt_len_mean,
            'Packet Length Std': pkt_len_std,
            'Packet Length Variance': pkt_len_var,
            'FIN Flag Count': float(self.fin_cnt),
            'SYN Flag Count': float(self.syn_cnt),
            'RST Flag Count': float(self.rst_cnt),
            'PSH Flag Count': float(self.psh_cnt),
            'ACK Flag Count': float(self.ack_cnt),
            'URG Flag Count': float(self.urg_cnt),
            'CWE Flag Count': 0.0,
            'ECE Flag Count': 0.0,
            'Down/Up Ratio': self.bwd_bytes / max(self.fwd_bytes, 1),
            'Average Packet Size': (self.fwd_bytes + self.bwd_bytes) / max(total, 1),
            'Avg Fwd Segment Size': self.fwd_bytes / max(self.fwd_pkts, 1),
            'Avg Bwd Segment Size': self.bwd_bytes / max(self.bwd_pkts, 1),
            'Fwd Header Length.1': 0.0,
            'Fwd Avg Bytes/Bulk': float(self.fwd_bytes),
            'Fwd Avg Packets/Bulk': float(self.fwd_pkts),
            'Fwd Avg Bulk Rate': float(self.fwd_bytes) / duration,
            'Bwd Avg Bytes/Bulk': float(self.bwd_bytes),
            'Bwd Avg Packets/Bulk': float(self.bwd_pkts),
            'Bwd Avg Bulk Rate': float(self.bwd_bytes) / duration,
            'Subflow Fwd Packets': float(self.fwd_pkts),
            'Subflow Fwd Bytes': float(self.fwd_bytes),
            'Subflow Bwd Packets': float(self.bwd_pkts),
            'Subflow Bwd Bytes': float(self.bwd_bytes),
            'Init_Win_bytes_forward': float(self.fwd_init_win),
            'Init_Win_bytes_backward': float(self.bwd_init_win),
            'act_data_pkt_fwd': float(self.act_data_pkt_fwd),
            'min_seg_size_forward': float(min(self.fwd_pkt_lens)) if self.fwd_pkt_lens else 0,
            'Active Mean': self.total_active / max(duration, 0.001),
            'Active Std': 0.0,
            'Active Max': self.total_active,
            'Active Min': 0.0,
            'Idle Mean': self.total_idle / max(duration, 0.001),
            'Idle Std': 0.0,
            'Idle Max': self.total_idle,
            'Idle Min': 0.0,
        }
        return raw

    def get_flow_info(self) -> dict:
        return {
            'src_ip': self.key[0], 'dst_ip': self.key[1],
            'src_port': self.key[2], 'dst_port': self.key[3],
            'protocol': {TCP_PROTO: 'TCP', UDP_PROTO: 'UDP'}.get(self.key[4], str(self.key[4])),
            'packet_length': self.fwd_bytes + self.bwd_bytes,
            'fwd_pkts': self.fwd_pkts, 'bwd_pkts': self.bwd_pkts,
        }


# ── Flow Extractor ───────────────────────────────────────────────────

class FlowExtractor:
    """Aggregates packets into flows, produces scaler-normalized 78-dim feature vectors."""

    def __init__(self, scaler_path: str = SCALER_PATH):
        self.normalizer = FeatureNormalizer(scaler_path)
        self._flows: Dict[tuple, FlowState] = {}
        self._completed: List[Tuple[np.ndarray, dict]] = []

    def _make_key(self, pkt) -> Optional[Tuple]:
        if IP not in pkt:
            return None
        ip = pkt[IP]
        if TCP in pkt:
            sp, dp = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sp, dp = pkt[UDP].sport, pkt[UDP].dport
        else:
            sp = dp = 0
        src, dst = ip.src, ip.dst
        # Normalize for bi-directional matching
        if src > dst or (src == dst and sp > dp):
            src, dst = dst, src
            sp, dp = dp, sp
        return (src, dst, sp, dp, ip.proto)

    def add_packet(self, pkt) -> Optional[Tuple[np.ndarray, dict]]:
        """Process packet, return (normalized_features, flow_info) if flow completed."""
        if IP not in pkt:
            return None
        key = self._make_key(pkt)
        if key is None:
            return None

        pkt_time = time.time()
        # Determine direction
        src = pkt[IP].src
        direction = 'fwd' if (key[0] == src) else 'bwd'

        if key not in self._flows:
            if len(self._flows) >= MAX_FLOWS:
                oldest = min(self._flows, key=lambda k: self._flows[k].last_seen)
                self._flush_flow(oldest)
            self._flows[key] = FlowState(key, pkt[IP].proto, pkt_time)

        self._flows[key].update(pkt, direction, pkt_time)

        if self._flows[key].is_complete(pkt_time):
            return self._flush_flow(key)
        return None

    def _flush_flow(self, key: tuple) -> Optional[Tuple[np.ndarray, dict]]:
        flow = self._flows.pop(key, None)
        if flow is None:
            return None
        raw = flow.build_raw_vector()
        # Convert dict to array in scaler feature order
        raw_arr = np.array([raw.get(n, 0.0) for n in self.normalizer.feature_names], dtype=np.float32)
        norm = self.normalizer.normalize(raw_arr)
        return norm, flow.get_flow_info()

    def flush_all(self) -> List[Tuple[np.ndarray, dict]]:
        results = []
        for key in list(self._flows.keys()):
            r = self._flush_flow(key)
            if r:
                results.append(r)
        return results

    def get_completed(self) -> List[Tuple[np.ndarray, dict]]:
        results = list(self._completed)
        self._completed.clear()
        now = time.time()
        for k, f in list(self._flows.items()):
            if now - f.last_seen > FLOW_TIMEOUT:
                r = self._flush_flow(k)
                if r:
                    results.append(r)
        return results

    @property
    def active_flow_count(self) -> int:
        return len(self._flows)
