"""
Spider-Sense v2 - Flask Backend API
Multi-agent collaborative open-set network intrusion detection system.
"""
import os
import sys
import json
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add parent to path for packaging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.crosr_engine import CROSREngine
from backend.orchestrator import Orchestrator
from backend.rag_engine import RAGEngine
from backend.xai_engine import XAIEngine
from backend.capture_engine_2 import CaptureEngine, HAS_SCAPY

# ── Path resolution (works in both dev and PyInstaller) ────────────

def _get_base_dir():
    """Get app root dir — handles PyInstaller _MEIPASS."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = _get_base_dir()

# Static folder: dev = '../web-frontend/build', PyInstaller = 'web-frontend/build'
_static = os.path.join(BASE_DIR, 'web-frontend', 'build')
if not os.path.isdir(_static):
    _static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web-frontend', 'build')

app = Flask(__name__, static_folder=_static, static_url_path='')
CORS(app)

# ── Initialize components ──────────────────────────────────────────────

KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'knowledge')

# Detection Engine (CROSR: DHRNet + WeibullOpenMax)
engine = CROSREngine()
_model_dir = os.path.join(BASE_DIR, 'models', 'weibull_om')
try:
    engine.load(
        model_path=os.path.join(_model_dir, 'model.pth'),
        detector_path=os.path.join(_model_dir, 'detector.pkl'),
    )
    print("[✓] CROSR Detection Engine loaded")
except Exception as e:
    print(f"[!] CROSR Engine load failed: {e}")
    # Fallback: try old path
    try:
        engine.load(
            model_path='models/weibull_om/model.pth',
            detector_path='models/weibull_om/detector.pkl',
        )
        print("[✓] CROSR Engine loaded (fallback path)")
    except Exception as e2:
        print(f"[!] Fallback also failed: {e2}")

# RAG Engine — 优先从 detector 类质心构建，回退到 JSON 文件
rag = RAGEngine(knowledge_dir=KNOWLEDGE_DIR)
try:
    rag.build_from_detector(
        detector_path=os.path.join(_model_dir, 'detector.pkl'),
        label_names=engine.label_names,
    )
    rag.load_knowledge_base()  # 加载 MITRE ATT&CK 知识库
    print(f"[✓] RAG built from model centroids ({len(rag.attack_patterns)} patterns, "
          f"{len(rag.mitre_knowledge)} MITRE techniques)")
except Exception as e:
    print(f"[!] Falling back to JSON knowledge base: {e}")
    rag.load_knowledge_base()

# XAI Engine
xai = XAIEngine()

# Orchestrator
orchestrator = Orchestrator(engine=engine, rag=rag, xai=xai)

# Runtime LLM configuration
_llm_api_key = os.environ.get('LLM_API_KEY')
_llm_model = os.environ.get('LLM_MODEL', 'gpt-4')
_llm_api_base = os.environ.get('LLM_API_BASE', 'https://api.openai.com/v1')
_use_api = bool(_llm_api_key)


def _init_agent_layer():
    global _use_api, _llm_api_key, _llm_model, _llm_api_base
    _use_api = bool(_llm_api_key)
    if _llm_api_base:
        os.environ['LLM_API_BASE'] = _llm_api_base
    if _llm_model:
        os.environ['LLM_MODEL'] = _llm_model
    if _llm_api_key:
        os.environ['LLM_API_KEY'] = _llm_api_key
    elif 'LLM_API_KEY' in os.environ:
        del os.environ['LLM_API_KEY']

    orchestrator.init_agent_layer(
        use_api=_use_api,
        api_key=_llm_api_key,
        model_name=_llm_model,
    )


_init_agent_layer()
print(f"[✓] Agent layer initialized (mode: {'LLM' if _use_api else 'rule-based'})")

# ── Capture Engine (real packet capture) ─────────────────────────────
capture_engine = CaptureEngine(
    on_packet=lambda features, flow_info: _on_captured_packet(features, flow_info)
)
print(f"[{'✓' if capture_engine.available else '!'}] Capture engine: "
      f"{'scapy available' if capture_engine.available else 'scapy not installed (pip install scapy)'}")

# ── Captured packet callback & buffer ────────────────────────────────
_captured_results = []

def _on_captured_packet(features: np.ndarray, flow_info: dict):
    """Called by CaptureEngine for each captured packet."""
    try:
        result = orchestrator.process_flow(features, flow_info)
        _captured_results.append({
            'prediction': result['detection']['prediction'],
            'confidence': result['detection']['class_confidence'],
            'isUnknown': result['detection']['is_unknown'],
            'unknownScore': result['detection'].get('unknown_prob', 0),
            'srcIp': flow_info.get('src_ip', '0.0.0.0'),
            'dstIp': flow_info.get('dst_ip', '0.0.0.0'),
            'protocol': flow_info.get('protocol', 'UNKNOWN'),
            'timestamp': result['timestamp'],
            'features': features.tolist(),
            'rawResult': result,
            'sourceType': 'captured',
        })
    except Exception:
        pass


# ── Static files (React frontend) ─────────────────────────────────────

@app.route('/')
def serve_index():
    """Serve the React app."""
    static_dir = os.path.join(BASE_DIR, 'web-frontend', 'build')
    if os.path.exists(os.path.join(static_dir, 'index.html')):
        return send_from_directory(static_dir, 'index.html')
    return jsonify({
        'name': 'Spider-Sense v2 API',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': ['/api/health', '/api/detect', '/api/debate', '/api/rag/search', '/api/xai/explain'],
        'note': 'Frontend not built. Run: cd web-frontend && npm run build'
    })


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    static_dir = os.path.join(BASE_DIR, 'web-frontend', 'build')
    file_path = os.path.join(static_dir, path)
    if os.path.exists(file_path):
        return send_from_directory(static_dir, path)
    return serve_index()


# ── API Routes ─────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with component status."""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'engine': engine.get_stats(),
            'rag': rag.get_knowledge_stats(),
            'agent': {
                'mode': 'llm' if _use_api else 'rule_based',
                'initialized': orchestrator.agent_layer is not None,
            },
            'orchestrator': {
                'total_analyzed': orchestrator.total_count,
                'is_running': orchestrator.is_running,
            },
        },
    })


@app.route('/api/config/llm', methods=['GET'])
def get_llm_config():
    return jsonify({
        'success': True,
        'data': {
            'enabled': bool(_use_api),
            'model': _llm_model,
            'api_base': os.environ.get('LLM_API_BASE', 'https://api.openai.com/v1'),
        },
    })


@app.route('/api/config/llm', methods=['POST'])
def set_llm_config():
    global _llm_api_key, _llm_model, _llm_api_base
    data = request.json or {}
    _llm_api_key = data.get('api_key', _llm_api_key)
    _llm_model = data.get('model', _llm_model)
    _llm_api_base = data.get('api_base', _llm_api_base)
    _init_agent_layer()
    return jsonify({
        'success': True,
        'message': 'LLM configuration updated',
        'data': {
            'enabled': bool(_use_api),
            'model': _llm_model,
            'api_base': _llm_api_base,
        },
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Single detection endpoint.
    POST JSON: {"features": [f1, f2, ...], "flow_info": {...}}
    """
    data = request.json or {}
    features = data.get('features', [])

    if not features:
        return jsonify({'success': False, 'error': '缺少 features 字段'}), 400

    try:
        features = np.array(features, dtype=np.float32)
        flow_info = data.get('flow_info', {})

        # Full pipeline detection
        result = orchestrator.process_flow(features, flow_info)

        return jsonify({
            'success': True,
            'data': result,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/detect/batch', methods=['POST'])
def detect_batch():
    """
    Batch detection endpoint.
    POST JSON: {"samples": [{"features": [...], "flow_info": {...}}, ...]}
    """
    data = request.json or {}
    samples = data.get('samples', [])

    if not samples:
        return jsonify({'success': False, 'error': '缺少 samples 字段'}), 400

    try:
        results = []
        for sample in samples:
            features = np.array(sample.get('features', []), dtype=np.float32)
            flow_info = sample.get('flow_info', {})
            results.append(orchestrator.process_flow(features, flow_info))

        return jsonify({
            'success': True,
            'data': results,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/debate', methods=['POST'])
def debate():
    """
    Multi-agent debate endpoint.
    POST JSON: detection result + flow info
    """
    if not orchestrator.agent_layer:
        return jsonify({'success': False, 'error': 'Agent层未初始化'}), 503

    data = request.json or {}
    detection_result = data.get('detection', {})
    flow_info = data.get('flow_info', {})

    if not detection_result:
        return jsonify({'success': False, 'error': '缺少 detection 字段'}), 400

    try:
        # Run RAG search
        features = np.array(data.get('features', []), dtype=np.float32)
        rag_result = None
        if len(features) > 0:
            try:
                embedding = engine.extract_features(features)['embedding'][0]
                rag_result = rag.search(embedding)
            except Exception:
                rag_result = []

        # Run XAI
        xai_result = None
        if len(features) > 0:
            try:
                xai_result = xai.explain(features, engine)
            except Exception:
                xai_result = {}

        # Run agent debate
        result = orchestrator.agent_layer.debate(
            detection_result=detection_result,
            flow_info=flow_info,
            rag_context=rag_result,
            xai_context=xai_result,
        )

        return jsonify({
            'success': True,
            'data': {
                'debate': result,
                'rag': rag_result,
                'xai': xai_result,
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/analyze', methods=['POST'])
def analyze_event():
    """
    按需 Agent 分析 — 仅在用户点击「Agent深度分析」时触发。
    POST JSON: { features, flow_info, detection }
    不走自动管道，节省 token。
    """
    data = request.json or {}
    features = np.array(data.get('features', []), dtype=np.float32)
    flow_info = data.get('flow_info', {})
    detection_result = data.get('detection')

    if len(features) == 0:
        return jsonify({'success': False, 'error': '缺少 features'}), 400

    try:
        result = orchestrator.analyze(
            features=features,
            flow_info=flow_info,
            detection_result=detection_result,
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    """
    RAG knowledge base search.
    POST JSON: {"embedding": [...], "top_k": 5} or {"features": [...]}
    """
    data = request.json or {}
    embedding = data.get('embedding', None)
    features = data.get('features', None)
    top_k = data.get('top_k', 5)

    try:
        if embedding is None and features is not None:
            # Extract embedding from features
            features_arr = np.array(features, dtype=np.float32)
            model_output = engine.extract_features(features_arr)
            embedding = model_output['embedding'][0].tolist()

        if embedding is None:
            return jsonify({'success': False, 'error': '缺少 embedding 或 features 字段'}), 400

        results = rag.search(np.array(embedding), top_k=top_k)

        # Also search history
        history = rag.search_history(np.array(embedding), top_k=3)

        return jsonify({
            'success': True,
            'data': {
                'matches': results,
                'history_matches': history,
                'stats': rag.get_knowledge_stats(),
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/xai/explain', methods=['POST'])
def xai_explain():
    """
    XAI explanation endpoint.
    POST JSON: {"features": [...]}
    """
    data = request.json or {}
    features = data.get('features', [])

    if not features:
        return jsonify({'success': False, 'error': '缺少 features 字段'}), 400

    try:
        features_arr = np.array(features, dtype=np.float32)
        explanation = xai.explain(features_arr, engine)

        return jsonify({
            'success': True,
            'data': explanation,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get session statistics."""
    return jsonify({
        'success': True,
        'data': orchestrator.get_session_stats(),
    })


@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    """Reset session statistics."""
    orchestrator.reset_stats()
    return jsonify({'success': True, 'message': '统计已重置'})


@app.route('/api/knowledge/stats', methods=['GET'])
def get_knowledge_stats():
    """Get knowledge base statistics."""
    return jsonify({
        'success': True,
        'data': rag.get_knowledge_stats(),
    })


@app.route('/api/knowledge/mitre/<technique_id>', methods=['GET'])
def get_mitre_technique(technique_id):
    """Get MITRE ATT&CK technique details."""
    if not rag._loaded:
        rag.load_knowledge_base()

    technique = rag.mitre_knowledge.get(technique_id)
    if not technique:
        return jsonify({'success': False, 'error': '技术ID未找到'}), 404

    return jsonify({
        'success': True,
        'data': {
            'id': technique_id,
            **technique,
        },
    })


# ── Dataset results (static, from original CROSR) ────────────────────

DATASET_RESULTS = {
    'cicids': {
        'name': 'CICIDS 2017', 'auroc': 0.9651, 'aupr_out': 0.9482,
        'fpr95': 0.1243, 'precision': 0.8937, 'recall': 0.9357, 'f1': 0.9143,
        'balanced_accuracy': 0.9210, 'knownClasses': 6, 'totalSamples': 25000,
    },
    'unsw_nb15': {
        'name': 'UNSW-NB15', 'auroc': 0.8949, 'aupr_out': 0.6106,
        'fpr95': 0.3627, 'precision': 0.5843, 'recall': 0.8718, 'f1': 0.6997,
        'balanced_accuracy': 0.8396, 'knownClasses': 6, 'totalSamples': 35000,
    },
    'nslkdd': {
        'name': 'NSL-KDD', 'auroc': 0.7587, 'aupr_out': 0.2584,
        'fpr95': 0.6873, 'precision': 0.2823, 'recall': 0.7835, 'f1': 0.4148,
        'balanced_accuracy': 0.7417, 'knownClasses': 5, 'totalSamples': 20000,
    },
    'cicids2018': {
        'name': 'CICIDS 2018', 'auroc': 0.4613, 'aupr_out': 0.3446,
        'fpr95': 0.7100, 'precision': 0.4665, 'recall': 0.9957, 'f1': 0.6344,
        'balanced_accuracy': 0.6318, 'knownClasses': 8, 'totalSamples': 30000,
    },
}


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    return jsonify({'success': True, 'data': DATASET_RESULTS})


@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    if dataset_id not in DATASET_RESULTS:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404
    return jsonify({'success': True, 'data': DATASET_RESULTS[dataset_id]})

CAPTURE_ATTACK_TYPES = ['DDoS', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'Unknown Attack']
_CAPTURE_CACHE = {}


def _load_capture_samples():
    """Load CICIDS demo samples for injection and reuse them in memory."""
    if 'cicids' in _CAPTURE_CACHE:
        return _CAPTURE_CACHE['cicids']

    known_path = os.path.join(BASE_DIR, 'processed_cicids', 'val_known.npz')
    open_path = os.path.join(BASE_DIR, 'processed_cicids', 'open_set.npz')
    known = np.load(known_path, allow_pickle=True)
    open_set = np.load(open_path, allow_pickle=True)

    _CAPTURE_CACHE['cicids'] = {
        'known': {
            'x': known['x'],
            'y': known['y'],
            'labels': [str(l) for l in known['label_names'].tolist()],
        },
        'open': {
            'x': open_set['x'],
            'y': open_set['y'],
            'labels': [str(l) for l in open_set['label_names'].tolist()],
        },
    }
    return _CAPTURE_CACHE['cicids']


def _sample_capture_vectors(attack_type, count):
    samples = _load_capture_samples()

    if attack_type == 'Unknown Attack':
        x = samples['open']['x']
        y = samples['open']['y']
        labels = samples['open']['labels']
        if len(x) == 0:
            raise ValueError('Open-set 样本加载失败')

        indices = np.random.choice(len(x), size=min(count, len(x)), replace=False)
        return [(
            x[int(idx)].astype(np.float32),
            labels[int(y[int(idx)])],
        ) for idx in indices]

    labels = samples['known']['labels']
    if attack_type not in labels:
        raise ValueError(f'不支持的攻击类型: {attack_type}')

    class_idx = labels.index(attack_type)
    mask = samples['known']['y'] == class_idx
    selected = samples['known']['x'][mask]
    if len(selected) == 0:
        raise ValueError(f'无法获取类别 {attack_type} 的样本')

    indices = np.random.choice(len(selected), size=min(count, len(selected)), replace=False)
    return [(
        selected[int(idx)].astype(np.float32),
        attack_type,
    ) for idx in indices]

# ── Main ────────────────────────────────────────────────────────────────

def create_app():
    """Create and configure the Flask app."""
    return app


# ── Dataset Sampling for Frontend Simulation ─────────────────────────

@app.route('/api/sample', methods=['POST'])
def get_sample():
    """
    Return a random feature vector from the CICIDS dataset.
    POST JSON: {"type": "normal"|"known"|"unknown"}
    """
    data = request.json or {}
    stype = data.get('type', 'normal')
    samples = _load_capture_samples()

    try:
        if stype == 'unknown':
            x = samples['open']['x']
            y = samples['open']['y']
            labels = samples['open']['labels']
            idx = np.random.randint(0, len(x))
            return jsonify({
                'success': True,
                'data': {
                    'features': x[int(idx)].astype(np.float32).tolist(),
                    'true_label': labels[int(y[int(idx)])],
                    'source': 'dataset_open',
                }
            })
        else:
            x = samples['known']['x']
            y = samples['known']['y']
            # normal = BENIGN (class 0), known = any attack (class 1-5)
            if stype == 'normal':
                mask = y == 0
            else:  # 'known'
                mask = y >= 1
            selected_idx = np.where(mask)[0]
            if len(selected_idx) == 0:
                return jsonify({'success': False, 'error': '无匹配样本'}), 400
            idx = np.random.choice(selected_idx)
            label = samples['known']['labels'][int(y[idx])]
            return jsonify({
                'success': True,
                'data': {
                    'features': x[int(idx)].astype(np.float32).tolist(),
                    'true_label': label,
                    'source': 'dataset_known',
                }
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Capture API (delegates to orchestrator) ────────────────────────────

@app.route('/api/capture/start', methods=['POST'])
def start_capture():
    data = request.json or {}
    # 不传 interface 则 auto-detect；传 'lo' 也 auto-detect (Windows 无 lo)
    interface = data.get('interface', None)
    if interface == 'lo':
        interface = None

    if not capture_engine.available:
        return jsonify({
            'success': False,
            'error': 'scapy 未安装。请运行 pip install scapy 并在 Windows 上安装 Npcap。',
            'fix': 'pip install scapy',
        }), 400

    # 显示可用接口供调试
    ifaces = capture_engine.list_interfaces()
    success = capture_engine.start(interface=interface)
    if not success:
        return jsonify({
            'success': False,
            'error': capture_engine.error or '启动捕获失败',
            'available_interfaces': ifaces,
        }), 500

    orchestrator.is_running = True
    return jsonify({
        'success': True,
        'message': f'数据包捕获已启动 (接口: {capture_engine.interface})',
        'interface': capture_engine.interface,
        'available_interfaces': ifaces,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/capture/status', methods=['GET'])
def get_capture_status():
    engine_status = capture_engine.get_status()
    return jsonify({
        'success': True,
        'data': {
            **orchestrator.get_status(),
            'capture_engine': engine_status,
        }
    })


@app.route('/api/capture/stop', methods=['POST'])
def stop_capture():
    capture_engine.stop()
    orchestrator.is_running = False
    _captured_results.clear()
    return jsonify({'success': True, 'message': '捕获已停止', 'timestamp': datetime.now().isoformat()})


@app.route('/api/capture/packets', methods=['GET'])
def get_captured_packets():
    """Return captured packets since last poll and clear the buffer."""
    packets = list(_captured_results)
    _captured_results.clear()
    return jsonify({
        'success': True,
        'data': packets,
        'count': len(packets),
    })


@app.route('/api/capture/inject', methods=['POST'])
def inject_attack():
    data = request.json or {}
    attack_type = data.get('attack_type', 'DDoS')
    count = int(data.get('count', 20))

    if attack_type not in CAPTURE_ATTACK_TYPES:
        return jsonify({
            'success': False,
            'error': f'不支持的攻击类型: {attack_type}. 可选: {sorted(CAPTURE_ATTACK_TYPES)}'
        }), 400

    if not orchestrator.is_running:
        return jsonify({
            'success': False,
            'error': '捕获未运行，请先启动捕获会话'
        }), 400

    if count < 1 or count > 50:
        return jsonify({'success': False, 'error': 'count 必须在 1-50 之间'}), 400

    try:
        sampled_vectors = _sample_capture_vectors(attack_type, count)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    injected_events = []
    for features, true_label in sampled_vectors:
        flow_info = {
            'src_ip': '127.0.0.1',
            'dst_ip': '127.0.0.1',
            'protocol': 'TCP',
            'packet_length': 100,
            'true_label': true_label,
            'attack_type': attack_type,
        }
        result = orchestrator.process_flow(features, flow_info)
        injected_events.append({
            'prediction': result['detection']['prediction'],
            'confidence': result['detection']['class_confidence'],
            'isUnknown': result['detection']['is_unknown'],
            'unknownScore': result['detection'].get('unknown_prob', 0),
            'srcIp': flow_info['src_ip'],
            'dstIp': flow_info['dst_ip'],
            'protocol': flow_info['protocol'],
            'timestamp': result['timestamp'],
            'features': features.tolist(),
            'rawResult': result,
            'trueLabel': true_label,
            'attackType': attack_type,
        })

    return jsonify({
        'success': True,
        'message': f'已注入 {attack_type} ({len(injected_events)} 条样本)',
        'data': injected_events,
    })


@app.route('/api/capture/stats', methods=['GET'])
def get_session_stats():
    return jsonify({'success': True, 'data': orchestrator.get_session_stats()})

@app.route('/api/capture/reset-stats', methods=['POST'])
def reset_session_stats():
    orchestrator.reset_stats()
    return jsonify({'success': True})


# Also update _captured_results field name reference



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"""
╔══════════════════════════════════════════════════════════╗
║        🕸️  Spider-Sense v2  —  蜘蛛感应 IDS               ║
║  多智能体协同 · 开放集识别 · RAG知识增强 · XAI可解释      ║
╠══════════════════════════════════════════════════════════╣
║  API Server:  http://localhost:{port}                      ║
║  Health:      http://localhost:{port}/api/health           ║
║  Mode:        {'LLM Agent' if _use_api else 'Rule-Based Agent'}                    ║
╚══════════════════════════════════════════════════════════╝
""")
    app.run(host='0.0.0.0', port=port, debug=debug)
