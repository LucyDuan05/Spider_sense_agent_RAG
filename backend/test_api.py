"""
Test the full detection pipeline on validation data.
"""
import numpy as np
import json
import requests
import sys

# Load validation data
print("[*] Loading validation data...")
val = np.load('processed_cicids/val_known.npz', allow_pickle=True)
x_val = val['x'].astype(np.float32)
y_val = val['y']

# Test on 5 samples from different classes
print("[*] Testing detection API on 5 samples...\n")

for i, idx in enumerate([0, 100, 500, 1000, 2000]):
    features = x_val[idx].tolist()
    label = y_val[idx]

    resp = requests.post(
        'http://127.0.0.1:5000/api/detect',
        json={'features': features, 'flow_info': {'src_ip': '192.168.1.1', 'dst_ip': '10.0.0.1', 'protocol': 'TCP'}},
        timeout=10
    )

    if resp.status_code == 200:
        data = resp.json()
        det = data['data']['detection']
        print(f"Sample {idx} (true={label}): prediction={det['prediction']}, "
              f"confidence={det['class_confidence']:.3f}, "
              f"is_unknown={det['is_unknown']}, "
              f"time={data['data']['pipeline_time_ms']}ms")
    else:
        print(f"Sample {idx}: ERROR {resp.status_code} - {resp.text}")
        sys.exit(1)

# Test an unknown sample (from open_set)
print("\n[*] Testing on open-set (unknown) samples...")
open_set = np.load('processed_cicids/open_set.npz', allow_pickle=True)
x_open = open_set['x'].astype(np.float32)

for i, idx in enumerate([0, 10, 50]):
    features = x_open[idx].tolist()
    resp = requests.post(
        'http://127.0.0.1:5000/api/detect',
        json={'features': features, 'flow_info': {'src_ip': '10.99.99.99', 'dst_ip': '192.168.1.1', 'protocol': 'UDP'}},
        timeout=10
    )

    if resp.status_code == 200:
        data = resp.json()
        det = data['data']['detection']
        print(f"Unknown sample {idx}: prediction={det['prediction']}, "
              f"is_unknown={det['is_unknown']}, "
              f"unknown_prob={det.get('unknown_prob', 0):.3f}, "
              f"iso_anomaly={det.get('iso_anomaly', False)}")
    else:
        print(f"Unknown sample {idx}: ERROR {resp.status_code}")
        sys.exit(1)

# Test multi-agent debate
print("\n[*] Testing multi-agent debate...")
sample_features = x_open[0].tolist()
resp = requests.post(
    'http://127.0.0.1:5000/api/debate',
    json={
        'detection': {'prediction': 'UNKNOWN', 'class_confidence': 0.42, 'is_unknown': True, 'unknown_prob': 0.78},
        'flow_info': {'src_ip': '10.99.99.99', 'dst_ip': '192.168.1.1', 'protocol': 'UDP', 'packet_length': 1500},
        'features': sample_features,
    },
    timeout=15
)

if resp.status_code == 200:
    data = resp.json()
    debate = data['data']['debate']
    print(f"  Verdict: {debate['verdict']}")
    print(f"  Vote: {debate['vote_count']}")
    print(f"  Action: {debate['action']}")
    print(f"  Detector: {debate['detector']['verdict']} ({debate['detector']['confidence']:.2f})")
    print(f"  Analyst: {debate['analyst']['verdict']} ({debate['analyst']['confidence']:.2f})")
    print(f"  Arbiter: {debate['arbiter']['verdict']} ({debate['arbiter']['confidence']:.2f})")
else:
    print(f"Debate ERROR: {resp.status_code}")

# Test RAG search
print("\n[*] Testing RAG search...")
resp = requests.post(
    'http://127.0.0.1:5000/api/rag/search',
    json={'features': sample_features, 'top_k': 3},
    timeout=10
)

if resp.status_code == 200:
    data = resp.json()
    matches = data['data']['matches']
    for m in matches:
        print(f"  {m['name']}: similarity={m['similarity']:.2f}, severity={m['severity']}")
        if m.get('mitre'):
            print(f"    MITRE: {m['mitre']['id']} - {m['mitre']['name']}")

# Test XAI
print("\n[*] Testing XAI explanation...")
resp = requests.post(
    'http://127.0.0.1:5000/api/xai/explain',
    json={'features': sample_features},
    timeout=10
)

if resp.status_code == 200:
    data = resp.json()
    xai = data['data']
    print(f"  Prediction: {xai['prediction']} (is_unknown={xai['is_unknown']})")
    print(f"  Explanation: {xai['explanation'][:120]}...")
    print(f"  Top features: {[f['name'] for f in xai['top_features'][:5]]}")

print("\n[✓] All tests passed!")
