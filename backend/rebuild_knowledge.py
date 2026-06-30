"""
Rebuild RAG knowledge base from actual CROSR data.
Extracts embeddings from known attack samples as attack pattern vectors.
"""
import numpy as np
import json, sys, os
sys.path.insert(0, '.')

from backend.crosr_engine import CROSREngine

print("[1] Loading engine...")
engine = CROSREngine()
engine.load()

# Load all datasets
datasets = {
    'cicids': 'processed_cicids',
}

print("\n[2] Building attack pattern vectors from training data...")

# For each known class, extract representative embeddings
train = np.load('processed_cicids/train_known.npz', allow_pickle=True)
x_train = train['x'].astype(np.float32)
y_train = train['y']
label_names = list(train['label_names'])

print(f"    Train: {len(x_train)} samples, {len(label_names)} classes: {label_names}")

# Extract 10 random embeddings per class
patterns = []
mitre_map = {
    'BENIGN': {'mitre_id': 'N/A', 'description': 'Normal traffic pattern', 'severity': 'low', 'category': 'benign'},
    'DDoS': {'mitre_id': 'T1498', 'description': 'Distributed Denial of Service flood attack', 'severity': 'critical', 'category': 'ddos'},
    'DoS Hulk': {'mitre_id': 'T1498', 'description': 'HTTP flood DoS attack generating unique requests', 'severity': 'critical', 'category': 'ddos'},
    'PortScan': {'mitre_id': 'T1046', 'description': 'Network port scanning reconnaissance', 'severity': 'medium', 'category': 'reconnaissance'},
    'FTP-Patator': {'mitre_id': 'T1110', 'description': 'FTP brute force credential attack', 'severity': 'high', 'category': 'credential_access'},
    'SSH-Patator': {'mitre_id': 'T1110', 'description': 'SSH brute force credential attack', 'severity': 'high', 'category': 'credential_access'},
}

for cls_id in range(len(label_names)):
    mask = y_train == cls_id
    cls_samples = x_train[mask]
    if len(cls_samples) == 0:
        continue

    # Pick diverse samples (evenly spaced)
    indices = np.linspace(0, len(cls_samples)-1, min(8, len(cls_samples)), dtype=int)
    name = str(label_names[cls_id])
    info = mitre_map.get(name, {'mitre_id': 'T1204', 'description': f'Known attack: {name}', 'severity': 'high', 'category': 'attack'})

    for j, idx in enumerate(indices):
        out = engine.extract_features(cls_samples[idx])
        emb = out['embedding'][0].tolist()

        patterns.append({
            'id': f'{name}_{j}',
            'name': f'{name} Pattern {j+1}',
            'mitre_id': info['mitre_id'],
            'description': info['description'],
            'severity': info['severity'],
            'category': info['category'],
            'embedding': emb,
        })

# Also add open-set samples as "unknown threat" patterns
print("\n[3] Adding open-set patterns...")
opn = np.load('processed_cicids/open_set.npz', allow_pickle=True)
x_open = opn['x'].astype(np.float32)
indices = np.linspace(0, len(x_open)-1, min(20, len(x_open)), dtype=int)
for j, idx in enumerate(indices):
    out = engine.extract_features(x_open[idx])
    emb = out['embedding'][0].tolist()
    patterns.append({
        'id': f'unknown_threat_{j}',
        'name': 'Unknown Attack Pattern',
        'mitre_id': 'T1204',
        'description': 'Previously unseen attack pattern detected by open-set recognition',
        'severity': 'high',
        'category': 'unknown',
        'embedding': emb,
    })

print(f"    Total patterns: {len(patterns)}")

# Save
os.makedirs('knowledge', exist_ok=True)
with open('knowledge/attack_patterns.json', 'w') as f:
    json.dump(patterns, f)
print(f"\n[✓] Saved {len(patterns)} patterns to knowledge/attack_patterns.json")

# Test RAG
print("\n[4] Testing RAG search...")
from backend.rag_engine import RAGEngine
rag = RAGEngine()
rag.load_knowledge_base()

# Test with known sample
out = engine.extract_features(x_train[0])
results = rag.search(out['embedding'][0], top_k=3)
print(f"    Known BENIGN matches:")
for r in results:
    print(f"      {r['name']}: sim={r['similarity']:.2f}")

# Test with open-set sample
out_u = engine.extract_features(x_open[10])
results_u = rag.search(out_u['embedding'][0], top_k=3)
print(f"    Unknown sample matches:")
for r in results_u:
    print(f"      {r['name']}: sim={r['similarity']:.2f}")

print("\n[✓] Knowledge base rebuilt!")
