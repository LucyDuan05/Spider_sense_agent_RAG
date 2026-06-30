"""Quick test of CROSREngine directly."""
import numpy as np
import sys
sys.path.insert(0, '.')
from backend.crosr_engine import CROSREngine

engine = CROSREngine()
engine.load()

val = np.load('processed_cicids/val_known.npz', allow_pickle=True)

print("Testing 10 samples...")
for i in [0, 100, 500, 1000]:
    try:
        r = engine.predict(val['x'][i])
        print(f"  [{i}] OK: pred={r['prediction']}, conf={r['class_confidence']}, "
              f"unknown={r['is_unknown']}, score={r.get('unknown_score','?')}")
    except Exception as e:
        print(f"  [{i}] ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\nCheck result types:")
r = engine.predict(val['x'][0])
for k, v in r.items():
    print(f"  {k}: {type(v).__name__} = {v if not isinstance(v, dict) else '{...}'}")
