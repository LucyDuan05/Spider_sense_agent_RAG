# -*- coding: utf-8 -*-
import os
import sys
import threading
from datetime import datetime

import numpy as np
import torch
from scapy.all import sniff, IP, TCP, UDP

try:
    from torch.serialization import safe_globals
except ImportError:
    safe_globals = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DHR_Net_1D import DHRNet1D

MODEL_PATH = os.path.join(PROJECT_ROOT, 'save_models', 'cicids_1d', 'latest.pth')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'processed_cicids', 'scaler.npz')

DEFAULT_LABELS = ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'SSH-Patator']


def _safe_torch_load(path):
    if safe_globals is None:
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(path, map_location='cpu')

    with safe_globals(['numpy.core.multiarray._reconstruct']):
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(path, map_location='cpu')


class IDSInferenceEngine:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.initialized = False
        self.is_running = False
        self.latest_prediction = 'Waiting...'
        self.latest_flow_info = {}
        self.recent_results = []
        self.total_count = 0
        self.lock = threading.Lock()
        self.mean = None
        self.scale = None
        self.feature_names = []
        self.label_names = list(DEFAULT_LABELS)
        self.model = None

        try:
            self._load(model_path, scaler_path)
            self.initialized = True
            print(f"[IDS] Model loaded: {len(self.label_names)} classes, {len(self.feature_names)} features")
        except Exception as exc:
            print(f"[WARNING] Failed to load model: {exc}")
            self.latest_prediction = 'MODEL_NOT_LOADED'

    def _load(self, model_path, scaler_path):
        scaler = np.load(scaler_path, allow_pickle=True)
        self.mean = scaler['mean']
        self.scale = scaler['scale']
        self.feature_names = [str(name) for name in scaler['feature_names'].tolist()]

        checkpoint = _safe_torch_load(model_path)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        num_classes = int(checkpoint.get('num_classes', 6))
        input_channels = int(checkpoint.get('input_channels', 1))
        base_channels = int(checkpoint.get('base_channels', 64))
        hidden_dim = int(checkpoint.get('hidden_dim', 512))

        self.model = DHRNet1D(
            num_classes=num_classes,
            input_channels=input_channels,
            base_channels=base_channels,
            hidden_dim=hidden_dim,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.label_names = [str(x) for x in checkpoint.get('label_names', np.array(DEFAULT_LABELS)).tolist()]

    def stop(self):
        self.is_running = False

    def _build_feature_vector(self, pkt):
        raw = {
            'Destination Port': 0.0,
            'Flow Duration': 0.0,
            'Total Fwd Packets': 1.0,
            'Total Backward Packets': 0.0,
            'Total Length of Fwd Packets': float(len(pkt)),
            'Total Length of Bwd Packets': 0.0,
            'Fwd Packet Length Max': float(len(pkt)),
            'Fwd Packet Length Min': float(len(pkt)),
            'Fwd Packet Length Mean': float(len(pkt)),
            'Fwd Packet Length Std': 0.0,
            'Bwd Packet Length Max': 0.0,
            'Bwd Packet Length Min': 0.0,
            'Bwd Packet Length Mean': 0.0,
            'Bwd Packet Length Std': 0.0,
            'Flow Bytes/s': float(len(pkt)),
            'Flow Packets/s': 1.0,
            'Flow IAT Mean': 0.0,
            'Flow IAT Std': 0.0,
            'Flow IAT Max': 0.0,
            'Flow IAT Min': 0.0,
            'Fwd IAT Total': 0.0,
            'Fwd IAT Mean': 0.0,
            'Fwd IAT Std': 0.0,
            'Fwd IAT Max': 0.0,
            'Fwd IAT Min': 0.0,
            'Bwd IAT Total': 0.0,
            'Bwd IAT Mean': 0.0,
            'Bwd IAT Std': 0.0,
            'Bwd IAT Max': 0.0,
            'Bwd IAT Min': 0.0,
            'Fwd PSH Flags': 0.0,
            'Bwd PSH Flags': 0.0,
            'Fwd URG Flags': 0.0,
            'Bwd URG Flags': 0.0,
            'Fwd Header Length': 0.0,
            'Bwd Header Length': 0.0,
            'Fwd Packets/s': 1.0,
            'Bwd Packets/s': 0.0,
            'Min Packet Length': float(len(pkt)),
            'Max Packet Length': float(len(pkt)),
            'Packet Length Mean': float(len(pkt)),
            'Packet Length Std': 0.0,
            'Packet Length Variance': 0.0,
            'FIN Flag Count': 0.0,
            'SYN Flag Count': 0.0,
            'RST Flag Count': 0.0,
            'PSH Flag Count': 0.0,
            'ACK Flag Count': 0.0,
            'URG Flag Count': 0.0,
            'CWE Flag Count': 0.0,
            'ECE Flag Count': 0.0,
            'Down/Up Ratio': 0.0,
            'Average Packet Size': float(len(pkt)),
            'Avg Fwd Segment Size': float(len(pkt)),
            'Avg Bwd Segment Size': 0.0,
            'Fwd Header Length.1': 0.0,
            'Fwd Avg Bytes/Bulk': float(len(pkt)),
            'Fwd Avg Packets/Bulk': 1.0,
            'Fwd Avg Bulk Rate': float(len(pkt)),
            'Bwd Avg Bytes/Bulk': 0.0,
            'Bwd Avg Packets/Bulk': 0.0,
            'Bwd Avg Bulk Rate': 0.0,
            'Subflow Fwd Packets': 1.0,
            'Subflow Fwd Bytes': float(len(pkt)),
            'Subflow Bwd Packets': 0.0,
            'Subflow Bwd Bytes': 0.0,
            'Init_Win_bytes_forward': 0.0,
            'Init_Win_bytes_backward': 0.0,
            'act_data_pkt_fwd': float(len(pkt)),
            'min_seg_size_forward': float(len(pkt)),
            'Active Mean': 0.0,
            'Active Std': 0.0,
            'Active Max': 0.0,
            'Active Min': 0.0,
            'Idle Mean': 0.0,
            'Idle Std': 0.0,
            'Idle Max': 0.0,
            'Idle Min': 0.0,
        }

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            raw['Destination Port'] = float(tcp.dport or 0)
            raw['Fwd PSH Flags'] = 1.0 if tcp.flags & 0x08 else 0.0
            raw['Fwd URG Flags'] = 1.0 if tcp.flags & 0x20 else 0.0
            raw['Fwd Header Length'] = float(pkt[IP].ihl * 4 + tcp.dataofs * 4)
            raw['Init_Win_bytes_forward'] = float(tcp.window or 0)
            raw['FIN Flag Count'] = 1.0 if tcp.flags & 0x01 else 0.0
            raw['SYN Flag Count'] = 1.0 if tcp.flags & 0x02 else 0.0
            raw['RST Flag Count'] = 1.0 if tcp.flags & 0x04 else 0.0
            raw['PSH Flag Count'] = 1.0 if tcp.flags & 0x08 else 0.0
            raw['ACK Flag Count'] = 1.0 if tcp.flags & 0x10 else 0.0
            raw['URG Flag Count'] = 1.0 if tcp.flags & 0x20 else 0.0
            raw['ECE Flag Count'] = 1.0 if tcp.flags & 0x40 else 0.0
            raw['Fwd Header Length.1'] = raw['Fwd Header Length']
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            raw['Destination Port'] = float(udp.dport or 0)
            raw['Fwd Header Length'] = float(pkt[IP].ihl * 4 + 8)
            raw['Fwd Header Length.1'] = raw['Fwd Header Length']

        vector = np.zeros(len(self.feature_names), dtype=np.float32)
        for index, name in enumerate(self.feature_names):
            vector[index] = float(raw.get(name, 0.0))

        return vector

    def predict_features(self, features):
        if not self.initialized:
            raise RuntimeError('模型未初始化')
        arr = np.asarray(features, dtype=np.float32)
        if arr.shape[0] != len(self.feature_names):
            raise ValueError(f'features length must be {len(self.feature_names)}, got {arr.shape[0]}')

        x_norm = (arr - self.mean) / (self.scale + 1e-8)
        input_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence, pred = torch.max(probs, dim=0)

        label = self.label_names[pred.item()] if pred.item() < len(self.label_names) else f'CLASS_{pred.item()}'
        if float(confidence.item()) < 0.65:
            label = 'UNKNOWN'

        return {
            'prediction': label,
            'confidence': float(confidence.item()),
            'class_id': int(pred.item()),
            'probabilities': [float(x) for x in probs.tolist()]
        }

    def predict_packet(self, pkt):
        features = self._build_feature_vector(pkt)
        return self.predict_features(features)

    def predict_from_dict(self, data):
        if 'features' in data:
            return self.predict_features(data['features'])

        if 'src_ip' in data and 'dst_ip' in data and 'protocol' in data:
            pkt = IP(src=data.get('src_ip', '0.0.0.0'), dst=data.get('dst_ip', '0.0.0.0'))
            protocol = data.get('protocol', '').upper()
            if protocol == 'TCP':
                pkt = pkt / TCP(dport=int(data.get('dst_port', 0)), sport=int(data.get('src_port', 0)), flags=data.get('flags', 'S'))
            elif protocol == 'UDP':
                pkt = pkt / UDP(dport=int(data.get('dst_port', 0)), sport=int(data.get('src_port', 0)))
            return self.predict_packet(pkt)

        raise ValueError('packet 字段必须包含 src_ip、dst_ip 和 protocol')

    def _packet_callback(self, pkt):
        if not pkt.haslayer(IP):
            return

        result = self.predict_packet(pkt)
        now = datetime.now().isoformat()

        with self.lock:
            self.total_count += 1
            self.latest_prediction = result['prediction']
            self.latest_flow_info = {
                'src_ip': pkt[IP].src,
                'dst_ip': pkt[IP].dst,
                'protocol': 'TCP' if pkt.haslayer(TCP) else 'UDP' if pkt.haslayer(UDP) else 'OTHER',
                'confidence': result['confidence'],
                'prediction': result['prediction'],
                'timestamp': now,
                'packet_length': len(pkt),
                'src_port': int(pkt[TCP].sport if pkt.haslayer(TCP) else pkt[UDP].sport if pkt.haslayer(UDP) else 0),
                'dst_port': int(pkt[TCP].dport if pkt.haslayer(TCP) else pkt[UDP].dport if pkt.haslayer(UDP) else 0)
            }
            self.recent_results.insert(0, self.latest_flow_info.copy())
            if len(self.recent_results) > 50:
                self.recent_results.pop()

        print(f"[AI Alert] {self.latest_prediction} | {pkt[IP].src} -> {pkt[IP].dst}")

    def _predict_normalized(self, x_norm):
        """Run inference on already-normalized features (skips scaler step)."""
        input_tensor = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence, pred = torch.max(probs, dim=0)
        label = self.label_names[pred.item()] if pred.item() < len(self.label_names) else f'CLASS_{pred.item()}'
        if float(confidence.item()) < 0.65:
            label = 'UNKNOWN'
        return {'prediction': label, 'confidence': float(confidence.item())}

    def inject_attack_packets(self, attack_type, count=20):
        """Inject real feature vectors directly into recent_results.

        Known attacks (val_known.npz): model should classify correctly.
        'Unknown Attack' (open_set.npz): model should output UNKNOWN — demonstrating open-set recognition.
        """
        import time

        OPEN_SET_KEY = 'Unknown Attack'

        if attack_type == OPEN_SET_KEY:
            # Load open-set (novel/unseen attack) samples
            if not hasattr(self, '_open_set_data'):
                path = os.path.join(PROJECT_ROOT, 'processed_cicids', 'open_set.npz')
                d = np.load(path, allow_pickle=True)
                self._open_set_data = {'x': d['x'], 'y': d['y'], 'labels': [str(l) for l in d['label_names'].tolist()]}
            samples = self._open_set_data['x']
            indices = np.random.choice(len(samples), size=min(count, len(samples)), replace=False)
            true_labels = [self._open_set_data['labels'][self._open_set_data['y'][i]] for i in indices]
        else:
            # Load known-class samples
            if not hasattr(self, '_val_data'):
                val_path = os.path.join(PROJECT_ROOT, 'processed_cicids', 'val_known.npz')
                val = np.load(val_path, allow_pickle=True)
                self._val_data = {'x': val['x'], 'y': val['y'], 'labels': [str(l) for l in val['label_names'].tolist()]}
            labels = self._val_data['labels']
            if attack_type not in labels:
                print(f"[Inject] Unknown attack type: {attack_type}. Available: {labels}")
                return
            class_idx = labels.index(attack_type)
            mask = self._val_data['y'] == class_idx
            samples = self._val_data['x'][mask]
            indices = np.random.choice(len(samples), size=min(count, len(samples)), replace=False)
            true_labels = [attack_type] * len(indices)

        print(f"[Inject] Injecting {len(indices)} '{attack_type}' feature vectors")

        for i, idx in enumerate(indices):
            result = self._predict_normalized(samples[idx])
            now = datetime.now().isoformat()
            flow_info = {
                'src_ip': '127.0.0.1',
                'dst_ip': '127.0.0.1',
                'protocol': 'TCP',
                'confidence': result['confidence'],
                'prediction': result['prediction'],
                'timestamp': now,
                'packet_length': 100,
                'src_port': 55555,
                'dst_port': 80,
                'true_label': true_labels[i],  # For debug/logging only
            }
            with self.lock:
                self.total_count += 1
                self.latest_prediction = result['prediction']
                self.latest_flow_info = flow_info.copy()
                self.recent_results.insert(0, flow_info)
                if len(self.recent_results) > 50:
                    self.recent_results.pop()
            print(f"[Inject] {i+1}/{len(indices)} true={true_labels[i]} -> pred={result['prediction']} ({result['confidence']:.2f})")
            time.sleep(0.3)  # 1 packet per 0.3s, interleaves naturally with real traffic

    def run_detection(self, interface='eth0'):
        self.is_running = True
        self.total_count = 0
        try:
            while self.is_running:
                sniff(iface=interface, prn=self._packet_callback, store=False, timeout=1)
        except PermissionError as exc:
            with self.lock:
                self.latest_prediction = 'PERMISSION_ERROR'
                self.latest_flow_info = {'error': '需要 root 权限来抓包，请以 sudo 运行后端服务器', 'timestamp': datetime.now().isoformat()}
        except Exception as exc:
            with self.lock:
                self.latest_prediction = 'ERROR'
                self.latest_flow_info = {'error': str(exc), 'timestamp': datetime.now().isoformat()}
        finally:
            self.is_running = False


engine = IDSInferenceEngine()
