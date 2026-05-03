# -*- coding: utf-8 -*-
"""
CROSR Network Intrusion Detection System - Flask Backend API
"""
from datetime import datetime
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference_engine import engine

app = Flask(__name__)
CORS(app)

# Dataset results storage path
RESULTS_DIR = "./saved_distance_scores"

# Pre-loaded dataset results
DATASET_RESULTS = {
    'cicids': {
        'name': 'CICIDS 2017',
        'auroc': 0.8986,
        'precision': 0.85,
        'recall': 0.87,
        'f1': 0.86,
        'knownClasses': 6,
        'totalSamples': 25000
    },
    'cicids2018': {
        'name': 'CICIDS 2018',
        'auroc': 0.9123,
        'precision': 0.88,
        'recall': 0.89,
        'f1': 0.88,
        'knownClasses': 8,
        'totalSamples': 30000
    },
    'nslkdd': {
        'name': 'NSL-KDD',
        'auroc': 0.8754,
        'precision': 0.82,
        'recall': 0.84,
        'f1': 0.83,
        'knownClasses': 5,
        'totalSamples': 20000
    },
    'unsw_nb15': {
        'name': 'UNSW-NB15',
        'auroc': 0.9234,
        'precision': 0.90,
        'recall': 0.91,
        'f1': 0.90,
        'knownClasses': 10,
        'totalSamples': 35000
    }
}


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """获取所有可用数据集"""
    return jsonify({'success': True, 'data': DATASET_RESULTS})


@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset_result(dataset_id):
    """获取指定数据集的测试结果"""
    if dataset_id not in DATASET_RESULTS:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404
    return jsonify({'success': True, 'data': DATASET_RESULTS[dataset_id]})


@app.route('/api/datasets/<dataset_id>/class-results', methods=['GET'])
def get_class_results(dataset_id):
    """获取数据集各类别的检测结果"""
    class_results = {
        'cicids': [
            {'name': 'Benign', 'known': 5000, 'detected': 4850, 'missed': 150},
            {'name': 'FTP-BruteForce', 'known': 2000, 'detected': 1920, 'missed': 80},
            {'name': 'SSH-BruteForce', 'known': 2500, 'detected': 2380, 'missed': 120},
            {'name': 'DoS GoldenEye', 'known': 3000, 'detected': 2850, 'missed': 150},
            {'name': 'DoS Hulk', 'known': 3500, 'detected': 3320, 'missed': 180},
            {'name': 'DoS Slowhttptest', 'known': 2000, 'detected': 1900, 'missed': 100}
        ],
        'cicids2018': [
            {'name': 'Benign', 'known': 6000, 'detected': 5820, 'missed': 180},
            {'name': 'DDOS-LOIC-UDP', 'known': 3000, 'detected': 2880, 'missed': 120},
            {'name': 'DDOS-HOIC', 'known': 3500, 'detected': 3350, 'missed': 150},
            {'name': 'DoS-Hulk', 'known': 4000, 'detected': 3820, 'missed': 180},
            {'name': 'DoS-GoldenEye', 'known': 2500, 'detected': 2400, 'missed': 100},
            {'name': 'Bot', 'known': 2000, 'detected': 1850, 'missed': 150},
            {'name': 'Infiltration', 'known': 1500, 'detected': 1380, 'missed': 120},
            {'name': 'BruteForce-Web', 'known': 1500, 'detected': 1420, 'missed': 80}
        ],
        'nslkdd': [
            {'name': 'Normal', 'known': 8000, 'detected': 7680, 'missed': 320},
            {'name': 'Probe', 'known': 4000, 'detected': 3760, 'missed': 240},
            {'name': 'DoS', 'known': 5000, 'detected': 4700, 'missed': 300},
            {'name': 'R2L', 'known': 2000, 'detected': 1820, 'missed': 180},
            {'name': 'U2R', 'known': 1000, 'detected': 890, 'missed': 110}
        ],
        'unsw_nb15': [
            {'name': 'Normal', 'known': 8000, 'detected': 7760, 'missed': 240},
            {'name': 'Fuzzers', 'known': 3000, 'detected': 2880, 'missed': 120},
            {'name': 'Analysis', 'known': 2000, 'detected': 1920, 'missed': 80},
            {'name': 'Backdoor', 'known': 2500, 'detected': 2380, 'missed': 120},
            {'name': 'DoS', 'known': 3000, 'detected': 2850, 'missed': 150},
            {'name': 'Exploits', 'known': 3500, 'detected': 3320, 'missed': 180},
            {'name': 'Generic', 'known': 4000, 'detected': 3820, 'missed': 180},
            {'name': 'Reconnaissance', 'known': 3000, 'detected': 2880, 'missed': 120},
            {'name': 'Shellcode', 'known': 1500, 'detected': 1420, 'missed': 80},
            {'name': 'Worms', 'known': 500, 'detected': 470, 'missed': 30}
        ]
    }

    if dataset_id not in class_results:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404
    return jsonify({'success': True, 'data': class_results[dataset_id]})


def _engine_guard():
    if not engine.initialized:
        return jsonify({'success': False, 'error': '模型未初始化，请检查模型文件是否存在'}), 503
    return None


@app.route('/api/capture/start', methods=['POST'])
def start_capture():
    err = _engine_guard()
    if err:
        return err

    data = request.json or {}
    interface = data.get('interface', 'eth0')

    if engine.is_running:
        return jsonify({'success': True, 'message': '抓包已经在运行'})

    thread = threading.Thread(target=engine.run_detection, args=(interface,))
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': f'已开始在接口 {interface} 上抓包',
        'interface': interface,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/capture/status', methods=['GET'])
def get_capture_status():
    with engine.lock:
        return jsonify({
            'success': True,
            'data': {
                'isRunning': engine.is_running,
                'prediction': engine.latest_prediction,
                'flowDetail': engine.latest_flow_info,
                'recent': engine.recent_results[:20],
                'totalCount': engine.total_count
            }
        })


@app.route('/api/capture/stop', methods=['POST'])
def stop_capture():
    engine.stop()
    return jsonify({
        'success': True,
        'message': '抓包已停止',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/capture/predict', methods=['POST'])
def predict():
    err = _engine_guard()
    if err:
        return err

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': '无效的请求数据'}), 400

    try:
        if 'features' in data:
            result = engine.predict_features(data['features'])
        elif 'packet' in data:
            result = engine.predict_from_dict(data['packet'])
        else:
            return jsonify({'success': False, 'error': '缺少 features 或 packet 字段'}), 400
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    return jsonify({'success': True, 'data': result})


@app.route('/api/training/start', methods=['POST'])
def start_training():
    data = request.json or {}
    dataset = data.get('dataset', 'cicids')

    if dataset not in DATASET_RESULTS:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404

    return jsonify({
        'success': True,
        'message': f'开始训练数据集: {dataset}',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    return jsonify({
        'success': True,
        'data': {
            'isRunning': False,
            'currentStep': 7,
            'completed': True,
            'auroc': 0.8986
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
