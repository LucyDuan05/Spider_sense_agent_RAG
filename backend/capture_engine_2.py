"""
Spider-Sense v2 - Real Packet Capture Engine v2
Per-packet feature extraction matching original inference_engine.py pattern.
Uses training scaler for proper (x - mean) / scale normalization.
"""
import os, sys, time, json, threading
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional, Callable

try:
    from scapy.all import sniff, IP, TCP, UDP, Raw
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

# Load scaler once at module level
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'processed_cicids', 'scaler.npz')
_scaler_data = None
_scaler_mean = None
_scaler_scale = None
_scaler_feature_names = []

def _load_scaler():
    global _scaler_data, _scaler_mean, _scaler_scale, _scaler_feature_names
    if _scaler_data is not None:
        return
    data = np.load(SCALER_PATH, allow_pickle=True)
    _scaler_mean = data['mean'].astype(np.float32)
    _scaler_scale = data['scale'].astype(np.float32)
    _scaler_scale = np.where(_scaler_scale < 1e-10, 1.0, _scaler_scale)
    _scaler_feature_names = [str(n) for n in data['feature_names'].tolist()]

_load_scaler()


def build_feature_vector(pkt) -> np.ndarray:
    """
    Build 78-dim feature vector from a single packet, matching CICIDS order.
    Same pattern as the original inference_engine._build_feature_vector.
    """
    pkt_len = float(len(pkt))
    raw = {name: 0.0 for name in _scaler_feature_names}

    raw['Destination Port'] = 0.0
    raw['Flow Duration'] = 0.0
    raw['Total Fwd Packets'] = 1.0
    raw['Total Backward Packets'] = 0.0
    raw['Total Length of Fwd Packets'] = pkt_len
    raw['Total Length of Bwd Packets'] = 0.0
    raw['Fwd Packet Length Max'] = pkt_len
    raw['Fwd Packet Length Min'] = pkt_len
    raw['Fwd Packet Length Mean'] = pkt_len
    raw['Fwd Packet Length Std'] = 0.0
    raw['Bwd Packet Length Max'] = 0.0
    raw['Bwd Packet Length Min'] = 0.0
    raw['Bwd Packet Length Mean'] = 0.0
    raw['Bwd Packet Length Std'] = 0.0
    raw['Flow Bytes/s'] = pkt_len
    raw['Flow Packets/s'] = 1.0
    raw['Flow IAT Mean'] = 0.0
    raw['Flow IAT Std'] = 0.0
    raw['Flow IAT Max'] = 0.0
    raw['Flow IAT Min'] = 0.0
    raw['Fwd IAT Total'] = 0.0
    raw['Fwd IAT Mean'] = 0.0
    raw['Fwd IAT Std'] = 0.0
    raw['Fwd IAT Max'] = 0.0
    raw['Fwd IAT Min'] = 0.0
    raw['Bwd IAT Total'] = 0.0
    raw['Bwd IAT Mean'] = 0.0
    raw['Bwd IAT Std'] = 0.0
    raw['Bwd IAT Max'] = 0.0
    raw['Bwd IAT Min'] = 0.0
    raw['Fwd PSH Flags'] = 0.0
    raw['Bwd PSH Flags'] = 0.0
    raw['Fwd URG Flags'] = 0.0
    raw['Bwd URG Flags'] = 0.0
    raw['Fwd Header Length'] = 0.0
    raw['Bwd Header Length'] = 0.0
    raw['Fwd Packets/s'] = 1.0
    raw['Bwd Packets/s'] = 0.0
    raw['Min Packet Length'] = pkt_len
    raw['Max Packet Length'] = pkt_len
    raw['Packet Length Mean'] = pkt_len
    raw['Packet Length Std'] = 0.0
    raw['Packet Length Variance'] = 0.0
    raw['FIN Flag Count'] = 0.0
    raw['SYN Flag Count'] = 0.0
    raw['RST Flag Count'] = 0.0
    raw['PSH Flag Count'] = 0.0
    raw['ACK Flag Count'] = 0.0
    raw['URG Flag Count'] = 0.0
    raw['CWE Flag Count'] = 0.0
    raw['ECE Flag Count'] = 0.0
    raw['Down/Up Ratio'] = 0.0
    raw['Average Packet Size'] = pkt_len
    raw['Avg Fwd Segment Size'] = pkt_len
    raw['Avg Bwd Segment Size'] = 0.0
    raw['Fwd Header Length.1'] = 0.0
    raw['Fwd Avg Bytes/Bulk'] = pkt_len
    raw['Fwd Avg Packets/Bulk'] = 1.0
    raw['Fwd Avg Bulk Rate'] = pkt_len
    raw['Bwd Avg Bytes/Bulk'] = 0.0
    raw['Bwd Avg Packets/Bulk'] = 0.0
    raw['Bwd Avg Bulk Rate'] = 0.0
    raw['Subflow Fwd Packets'] = 1.0
    raw['Subflow Fwd Bytes'] = pkt_len
    raw['Subflow Bwd Packets'] = 0.0
    raw['Subflow Bwd Bytes'] = 0.0
    raw['Init_Win_bytes_forward'] = 0.0
    raw['Init_Win_bytes_backward'] = 0.0
    raw['act_data_pkt_fwd'] = pkt_len
    raw['min_seg_size_forward'] = pkt_len
    raw['Active Mean'] = 0.0
    raw['Active Std'] = 0.0
    raw['Active Max'] = 0.0
    raw['Active Min'] = 0.0
    raw['Idle Mean'] = 0.0
    raw['Idle Std'] = 0.0
    raw['Idle Max'] = 0.0
    raw['Idle Min'] = 0.0

    if TCP in pkt:
        tcp = pkt[TCP]
        raw['Destination Port'] = float(tcp.dport)
        raw['Fwd PSH Flags'] = 1.0 if tcp.flags & 0x08 else 0.0
        raw['Fwd URG Flags'] = 1.0 if tcp.flags & 0x20 else 0.0
        raw['Fwd Header Length'] = float(pkt[IP].ihl * 4 + tcp.dataofs * 4)
        raw['Init_Win_bytes_forward'] = float(tcp.window)
        raw['FIN Flag Count'] = 1.0 if tcp.flags & 0x01 else 0.0
        raw['SYN Flag Count'] = 1.0 if tcp.flags & 0x02 else 0.0
        raw['RST Flag Count'] = 1.0 if tcp.flags & 0x04 else 0.0
        raw['PSH Flag Count'] = 1.0 if tcp.flags & 0x08 else 0.0
        raw['ACK Flag Count'] = 1.0 if tcp.flags & 0x10 else 0.0
        raw['URG Flag Count'] = 1.0 if tcp.flags & 0x20 else 0.0
        raw['ECE Flag Count'] = 1.0 if tcp.flags & 0x40 else 0.0
        raw['CWE Flag Count'] = 1.0 if tcp.flags & 0x80 else 0.0
        raw['Fwd Header Length.1'] = raw['Fwd Header Length']
    elif UDP in pkt:
        raw['Destination Port'] = float(pkt[UDP].dport)

    raw_arr = np.array([raw.get(n, 0.0) for n in _scaler_feature_names], dtype=np.float32)
    norm = (raw_arr - _scaler_mean) / _scaler_scale
    return np.clip(norm, -10, 10).astype(np.float32)


def extract_flow_info(pkt) -> dict:
    info = {'src_ip': '0.0.0.0', 'dst_ip': '0.0.0.0', 'protocol': 'UNKNOWN', 'packet_length': 0}
    if IP in pkt:
        info['src_ip'] = pkt[IP].src
        info['dst_ip'] = pkt[IP].dst
        info['protocol'] = {6: 'TCP', 17: 'UDP'}.get(pkt[IP].proto, str(pkt[IP].proto))
    if TCP in pkt:
        info['src_port'] = pkt[TCP].sport
        info['dst_port'] = pkt[TCP].dport
    elif UDP in pkt:
        info['src_port'] = pkt[UDP].sport
        info['dst_port'] = pkt[UDP].dport
    info['packet_length'] = len(pkt)
    return info


class CaptureEngine:
    """Per-packet capture engine with scaler-normalized features."""

    def __init__(self, on_packet: Optional[Callable] = None):
        self.on_packet = on_packet
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self.packet_queue = deque(maxlen=500)
        self.captured_count = 0
        self.interface = None
        self.error: Optional[str] = None

    @property
    def available(self) -> bool:
        return HAS_SCAPY

    @staticmethod
    def list_interfaces() -> list:
        if not HAS_SCAPY: return []
        try:
            from scapy.all import get_working_ifaces
            return [{'name': i.name, 'ip': i.ip or '0.0.0.0', 'description': i.description}
                    for i in get_working_ifaces() if i.ip and i.ip != '0.0.0.0']
        except: return []

    @staticmethod
    def auto_select_interface() -> str:
        ifaces = CaptureEngine.list_interfaces()
        def _s(i):
            n, ip = i['name'].lower(), i['ip']; sc = 0
            if 'wlan' in n or 'wifi' in n: sc += 100
            if 'eth' in n or 'ethernet' in n: sc += 90
            if 'loopback' in n: sc -= 50
            if 'bluetooth' in n: sc -= 30
            if 'virtual' in n: sc -= 20
            if ip and ip != '0.0.0.0' and not ip.startswith('169.254'): sc += 50
            return sc
        scored = sorted([(i, _s(i)) for i in ifaces], key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > 0: return scored[0][0]['name']
        for i in ifaces:
            if i['ip'] != '0.0.0.0': return i['name']
        return ''

    def start(self, interface: str = None):
        if not HAS_SCAPY:
            self.error = 'scapy not installed'; return False
        if self._running: return True
        if not interface or interface == 'lo': interface = self.auto_select_interface()
        if not interface: self.error = 'no interface'; return False
        self.interface = interface
        self._stop_event.clear()
        self._running = True
        self.error = None
        def _sniff():
            try:
                sniff(iface=interface, prn=self._handle_packet, store=False,
                      stop_filter=lambda _: self._stop_event.is_set())
            except Exception as e:
                self.error = f'{type(e).__name__}: {e}'
                self._running = False
        self._thread = threading.Thread(target=_sniff, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    def _handle_packet(self, pkt):
        try:
            if IP not in pkt: return
            self.captured_count += 1
            features = build_feature_vector(pkt)
            flow_info = extract_flow_info(pkt)
            self.packet_queue.append({
                'features': features.tolist(),
                'flow_info': flow_info,
                'timestamp': datetime.now().isoformat(),
            })
            if self.on_packet:
                self.on_packet(features, flow_info)
        except Exception:
            pass

    def get_pending_packets(self, max_count: int = 50) -> list:
        pkts = []
        while self.packet_queue and len(pkts) < max_count:
            pkts.append(self.packet_queue.popleft())
        return pkts

    def get_status(self) -> dict:
        return {
            'available': self.available, 'running': self._running,
            'captured_count': self.captured_count,
            'queue_size': len(self.packet_queue),
            'interface': self.interface, 'error': self.error,
            'scapy_installed': HAS_SCAPY,
            'available_interfaces': self.list_interfaces(),
        }
