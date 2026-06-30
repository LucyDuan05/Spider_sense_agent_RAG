"""
Quick fix and test for RAG and XAI with CROSR engine.
"""
import numpy as np
import json
import sys
sys.path.insert(0, '.')

from backend.crosr_engine import CROSREngine
from backend.rag_engine import RAGEngine
from backend.xai_engine import XAIEngine

# Check feature names
print("[1] Checking feature metadata...")
with open('processed_cicids/metadata.json') as f:
    meta = json.load(f)
feature_names = meta.get('feature_names', meta.get('features', []))
print(f"    Input features: {len(feature_names)}")
for i, name in enumerate(feature_names[:5]):
    print(f"    [{i}] {name}")

# Load engine
print("\n[2] Loading engine...")
engine = CROSREngine()
engine.load()

# Test RAG with CROSR features
print("\n[3] Testing RAG with CROSR embedding...")
val = np.load('processed_cicids/val_known.npz', allow_pickle=True)
opn = np.load('processed_cicids/open_set.npz', allow_pickle=True)

rag = RAGEngine()
rag.load_knowledge_base()

# Get embedding from engine
out = engine.extract_features(val['x'][0])
emb = out['embedding'][0]
print(f"    Embedding dim: {len(emb)}")

results = rag.search(emb, top_k=3)
print(f"    RAG matches: {len(results)}")
for r in results:
    print(f"      {r['name']}: sim={r['similarity']:.2f}, severity={r['severity']}")

# Test with unknown sample
out_u = engine.extract_features(opn['x'][0])
emb_u = out_u['embedding'][0]
results_u = rag.search(emb_u, top_k=3)
print(f"    Unknown RAG matches: {len(results_u)}")

# Test XAI
print("\n[4] Testing XAI...")
xai = XAIEngine()

# Update feature names to match CICIDS if needed
if feature_names and len(feature_names) > len(xai.feature_names):
    xai.feature_names = list(feature_names) + xai.feature_names[len(feature_names):]
    print(f"    Updated XAI feature names: {len(xai.feature_names)}")

result = xai.explain(val['x'][0], engine)
print(f"    Prediction: {result['prediction']}")
print(f"    Top features: {[f['name'] for f in result['top_features'][:5]]}")
print(f"    Explanation: {result['explanation'][:100]}...")

print("\n[✓] RAG and XAI working!")
