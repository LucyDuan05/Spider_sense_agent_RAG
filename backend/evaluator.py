"""
Spider-Sense v2 — 消融实验评测框架
量化评估每个组件（阈值、OpenMax、RAG、反馈RAG）对开放集检测的贡献。

实验组:
  1. baseline + threshold       — 纯 Softmax 阈值判定未知
  2. baseline + OpenMax         — DHRNet + WeibullOpenMax
  3. baseline + RAG             — DHRNet 特征 + RAG 质心检索
  4. baseline + OpenMax + RAG   — 当前完整管道
  5. baseline + OpenMax + RAG   — 完整管道 + 反馈学习（更新知识库）
    + feedback

用法:
  python -c "from backend.evaluator import EvalRunner; EvalRunner().run_all()"
"""
import os
import sys
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             f1_score, precision_score, recall_score,
                             precision_recall_curve, auc)


# ── 路径 ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_cicids')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'weibull_om')
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'knowledge')
OUTPUT_DIR = os.path.join(BASE_DIR, 'eval_results')


@dataclass
class EvalConfig:
    """单一实验配置."""
    name: str                           # 实验代号
    use_openmax: bool = True            # WeibullOpenMax
    use_rag: bool = False               # RAG 检索
    use_rag_feedback: bool = False      # 反馈更新 RAG 知识库
    threshold_confidence: float = 0.7   # Softmax 置信度阈值
    threshold_unknown: float = 0.5      # OpenMax unknown 阈值
    threshold_rag_sim: float = 0.6      # RAG 相似度阈值
    max_samples: int = 0                # 0=全量; N=每类最多N条
    seed: int = 42
    description: str = ''               # 实验描述


# ── 所有实验定义 ──────────────────────────────────────────────────
EXPERIMENTS = [
    EvalConfig(
        name='baseline_threshold',
        use_openmax=False,
        use_rag=False,
        use_rag_feedback=False,
        description='纯 Softmax 阈值: max(softmax) < 0.7 → unknown',
    ),
    EvalConfig(
        name='baseline_openmax',
        use_openmax=True,
        use_rag=False,
        use_rag_feedback=False,
        description='DHRNet + WeibullOpenMax 开放集判定',
    ),
    EvalConfig(
        name='baseline_rag',
        use_openmax=False,
        use_rag=True,
        use_rag_feedback=False,
        description='DHRNet 特征 + RAG 质心余弦相似度检索',
    ),
    EvalConfig(
        name='full_openmax_rag',
        use_openmax=True,
        use_rag=True,
        use_rag_feedback=False,
        description='OpenMax + RAG 完整管道',
    ),
    EvalConfig(
        name='full_openmax_rag_feedback',
        use_openmax=True,
        use_rag=True,
        use_rag_feedback=True,
        description='完整管道 + 反馈学习: 检测失误自动更新 RAG',
    ),
]


class EvalRunner:
    """评测运行器: 加载数据 → 逐实验检测 → 计算指标 → 输出报告."""

    def __init__(self):
        self.label_names = ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan',
                            'FTP-Patator', 'SSH-Patator']
        self.num_classes = len(self.label_names)

        # ── 延迟加载 ──
        self.engine = None
        self.rag = None
        self._data_loaded = False

        # ── 数据 ──
        self.known_features = None   # [N, 78]
        self.known_labels = None     # [N]
        self.unknown_features = None # [M, 78]

        # ── 输出 ──
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ───────────────────────── 数据加载 ─────────────────────────

    def load_data(self, max_samples: int = 0):
        """加载 val_known + open_set."""
        val = np.load(os.path.join(DATA_DIR, 'val_known.npz'), allow_pickle=True)
        opn = np.load(os.path.join(DATA_DIR, 'open_set.npz'), allow_pickle=True)

        x_known, y_known = val['x'].astype(np.float32), val['y']
        x_unknown = opn['x'].astype(np.float32)

        print(f"[数据] Known: {x_known.shape}, Unknown: {x_unknown.shape}")

        if max_samples > 0:
            # 每类分层抽样
            idx_list = []
            for cls in range(self.num_classes):
                cls_idx = np.where(y_known == cls)[0]
                n = min(max_samples, len(cls_idx))
                rng = np.random.RandomState(42)
                chosen = rng.choice(cls_idx, n, replace=False)
                idx_list.append(chosen)
            idx_known = np.concatenate(idx_list)
            x_known = x_known[idx_known]
            y_known = y_known[idx_known]

            n_unk = min(max_samples * 2, len(x_unknown))
            rng = np.random.RandomState(42)
            idx_unk = rng.choice(len(x_unknown), n_unk, replace=False)
            x_unknown = x_unknown[idx_unk]
            print(f"[数据] 降采样后: Known={x_known.shape}, Unknown={x_unknown.shape}")

        self.known_features = x_known
        self.known_labels = y_known
        self.unknown_features = x_unknown
        self._data_loaded = True

    def load_engine(self):
        """加载 CROSR Engine + RAG."""
        sys.path.insert(0, BASE_DIR)

        from backend.crosr_engine import CROSREngine
        from backend.rag_engine import RAGEngine

        # Engine
        self.engine = CROSREngine()
        self.engine.load(
            model_path=os.path.join(MODEL_DIR, 'model.pth'),
            detector_path=os.path.join(MODEL_DIR, 'detector.pkl'),
        )

        # RAG
        self.rag = RAGEngine(knowledge_dir=KNOWLEDGE_DIR)
        self.rag.build_from_detector(
            detector_path=os.path.join(MODEL_DIR, 'detector.pkl'),
            label_names=self.label_names,
        )
        self.rag.load_knowledge_base()

        print(f"[引擎] CROSR + RAG ({len(self.rag.attack_patterns)} patterns) loaded")

    # ───────────────────────── 单样本检测 ──────────────────────

    def _predict_threshold(self, features: np.ndarray, conf_thresh: float) -> Dict:
        """仅用 Softmax 阈值判定未知."""
        out = self.engine.extract_features(features)
        probs = out['probabilities'][0]
        max_prob = float(np.max(probs))
        pred_class = int(np.argmax(probs))

        if max_prob >= conf_thresh:
            return {
                'prediction': self.label_names[pred_class],
                'predicted_class': pred_class,
                'class_confidence': max_prob,
                'is_unknown': False,
                'unknown_score': 1.0 - max_prob,
                'embedding': out['embedding'][0],
            }
        else:
            return {
                'prediction': 'UNKNOWN',
                'predicted_class': -1,
                'class_confidence': max_prob,
                'is_unknown': True,
                'unknown_score': 1.0 - max_prob,
                'embedding': out['embedding'][0],
            }

    def _predict_openmax(self, features: np.ndarray) -> Dict:
        """DHRNet + OpenMax 检测 (用 return_details 获取 unknown_score)."""
        return self.engine.predict(features, return_details=True)

    def _predict_rag(self, features: np.ndarray, sim_thresh: float) -> Dict:
        """DHRNet 特征 + RAG 质心检索."""
        out = self.engine.extract_features(features)
        embedding = out['embedding'][0]
        probs = out['probabilities'][0]

        rag_results = self.rag.search(embedding, top_k=3, min_similarity=0.0)
        top_sim = rag_results[0]['similarity'] if rag_results else 0.0

        if top_sim >= sim_thresh:
            top = rag_results[0]
            name = top['name']
            if name in self.label_names:
                cls_id = self.label_names.index(name)
            else:
                cls_id = -1
            return {
                'prediction': name,
                'predicted_class': cls_id,
                'class_confidence': float(np.max(probs)),
                'is_unknown': False,
                'unknown_score': round(1.0 - top_sim, 4),
                'rag_similarity': top_sim,
                'rag_result': rag_results,
                'embedding': embedding,
            }
        else:
            return {
                'prediction': 'UNKNOWN',
                'predicted_class': -1,
                'class_confidence': float(np.max(probs)),
                'is_unknown': True,
                'unknown_score': round(1.0 - top_sim, 4),
                'rag_similarity': top_sim,
                'rag_result': rag_results,
                'embedding': embedding,
            }

    def _predict_full(self, features: np.ndarray, config: EvalConfig) -> Dict:
        """OpenMax + RAG 完整管道 + 融合 unknown_score."""
        det = self.engine.predict(features, return_details=False)
        embedding = np.array(det.get('embedding', []), dtype=np.float32)

        # RAG enrichment + 融合 unknown_score
        rag_result = None
        rag_sim = 0.0
        if config.use_rag and len(embedding) > 0:
            rag_result = self.rag.search(embedding, top_k=3)
            if rag_result:
                rag_sim = rag_result[0]['similarity']

        det['rag_result'] = rag_result

        # 融合判定
        om_unknown = det['is_unknown']
        rag_low = rag_sim < config.threshold_rag_sim

        # 综合 unknown_score: 取 OpenMax 和 1-RAG_sim 的加权最大值
        om_score = det.get('unknown_score', 0.5)
        rag_unknown_score = 1.0 - rag_sim
        combined_unknown = max(om_score * 0.6, rag_unknown_score * 0.4) if rag_result else om_score
        # 如果两个信号都报警，提高分数
        if om_unknown and rag_low:
            combined_unknown = max(combined_unknown, 0.9)
        elif om_unknown or rag_low:
            combined_unknown = max(combined_unknown, 0.7)

        det['unknown_score'] = round(float(combined_unknown), 4)
        det['is_unknown'] = combined_unknown > config.threshold_unknown
        if det['is_unknown']:
            det['prediction'] = 'UNKNOWN'
            det['predicted_class'] = -1

        return det

    # ───────────────────────── 实验运行 ──────────────────────

    def run_experiment(self, config: EvalConfig) -> Dict:
        """运行单个实验配置."""
        print(f"\n{'='*60}")
        print(f"[实验] {config.name}")
        print(f"       {config.description}")
        print(f"{'='*60}")

        if not self._data_loaded:
            self.load_data(config.max_samples)
        if self.engine is None:
            self.load_engine()

        rng = np.random.RandomState(config.seed)
        t0 = time.time()

        known_preds = []   # 已知类样本的预测结果
        unknown_preds = [] # 未知类样本的预测结果

        # ── 处理已知类 ──
        n_known = len(self.known_features)
        print(f"  已知类 ({n_known} samples)...", end=' ', flush=True)
        for i in range(n_known):
            feat = self.known_features[i]
            if config.name == 'baseline_threshold':
                pred = self._predict_threshold(feat, config.threshold_confidence)
            elif config.name == 'baseline_openmax':
                pred = self._predict_openmax(feat)
            elif config.name == 'baseline_rag':
                pred = self._predict_rag(feat, config.threshold_rag_sim)
            else:  # full_*
                pred = self._predict_full(feat, config)

            known_preds.append(pred)

            # ── 反馈学习 ──
            if config.use_rag_feedback:
                true_label = int(self.known_labels[i])
                pred_label = pred.get('predicted_class', -1)
                if pred_label != true_label and pred_label >= 0:
                    # 检测失误 → 把正确类别的质心加入 RAG
                    true_name = self.label_names[true_label]
                    # 找到对应质心的 pattern_id
                    for pid, meta in self.rag.attack_metadata.items():
                        if meta.get('name') == true_name and meta.get('source') == 'model_centroid':
                            emb = np.array(pred.get('embedding', []), dtype=np.float32)
                            if len(emb) > 0:
                                # 以当前样本的 embedding 作为新 pattern 加入
                                fb_id = f'feedback_{true_name}_{i}'
                                self.rag.attack_patterns[fb_id] = emb
                                self.rag.attack_metadata[fb_id] = dict(meta)
                                self.rag.attack_metadata[fb_id]['source'] = 'feedback'
                            break

        print(f"done")

        # ── 处理未知类 ──
        n_unknown = len(self.unknown_features)
        print(f"  未知类 ({n_unknown} samples)...", end=' ', flush=True)
        for i in range(n_unknown):
            feat = self.unknown_features[i]
            if config.name == 'baseline_threshold':
                pred = self._predict_threshold(feat, config.threshold_confidence)
            elif config.name == 'baseline_openmax':
                pred = self._predict_openmax(feat)
            elif config.name == 'baseline_rag':
                pred = self._predict_rag(feat, config.threshold_rag_sim)
            else:
                pred = self._predict_full(feat, config)

            unknown_preds.append(pred)
        print(f"done")

        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.1f}s ({elapsed/(n_known+n_unknown)*1000:.1f} ms/sample)")

        # ── 计算指标 ──
        metrics = self._compute_metrics(known_preds, self.known_labels, unknown_preds, config)

        # 如果跑反馈实验, 记录最终知识库大小
        if config.use_rag_feedback:
            n_fb = sum(1 for m in self.rag.attack_metadata.values()
                       if m.get('source') == 'feedback')
            metrics['feedback_added'] = n_fb
            metrics['rag_total_patterns'] = len(self.rag.attack_patterns)

        return {
            'config': {
                'name': config.name,
                'use_openmax': config.use_openmax,
                'use_rag': config.use_rag,
                'use_rag_feedback': config.use_rag_feedback,
                'threshold_confidence': config.threshold_confidence,
                'threshold_unknown': config.threshold_unknown,
                'threshold_rag_sim': config.threshold_rag_sim,
                'max_samples': config.max_samples or 'all',
            },
            'metrics': metrics,
            'timing': {
                'elapsed_seconds': round(elapsed, 2),
                'samples_per_second': round((n_known + n_unknown) / elapsed, 1),
            },
            'data_sizes': {
                'known': n_known,
                'unknown': n_unknown,
            },
        }

    # ───────────────────────── 指标计算 ──────────────────────

    def _compute_metrics(self, known_preds: List[Dict],
                         true_known_labels: np.ndarray,
                         unknown_preds: List[Dict],
                         config: EvalConfig) -> Dict:
        """计算开放集检测指标."""
        # ── 二分类开放集检测: known=0, unknown=1 ──
        y_true = np.array([0] * len(known_preds) + [1] * len(unknown_preds))
        unknown_scores = (
            [p.get('unknown_score', 0.5) for p in known_preds] +
            [p.get('unknown_score', 0.5) for p in unknown_preds]
        )
        unknown_scores = np.array(unknown_scores, dtype=np.float64)

        # 处理 NaN/Inf
        unknown_scores = np.nan_to_num(unknown_scores, nan=0.5, posinf=1.0, neginf=0.0)

        # Predicted labels (0=known, 1=unknown)
        y_pred = np.array(
            [1 if p['is_unknown'] else 0 for p in known_preds] +
            [1 if p['is_unknown'] else 0 for p in unknown_preds]
        )

        metrics = {}

        # AUROC — 开放集检测核心指标
        try:
            metrics['auroc'] = round(float(roc_auc_score(y_true, unknown_scores)), 4)
        except Exception:
            metrics['auroc'] = 0.0

        # AUPR (Out)
        try:
            precision_out, recall_out, _ = precision_recall_curve(y_true, unknown_scores)
            metrics['aupr_out'] = round(float(auc(recall_out, precision_out)), 4)
        except Exception:
            metrics['aupr_out'] = 0.0

        # FPR@95% TPR
        try:
            fpr, tpr, _ = roc_curve(y_true, unknown_scores)
            target_tpr = 0.95
            idx = np.argmin(np.abs(tpr - target_tpr))
            metrics['fpr_at_95tpr'] = round(float(fpr[idx]), 4)
        except Exception:
            metrics['fpr_at_95tpr'] = -1.0

        # TPR@5% FPR
        try:
            target_fpr = 0.05
            idx = np.argmin(np.abs(fpr - target_fpr))
            metrics['tpr_at_5fpr'] = round(float(tpr[idx]), 4)
        except Exception:
            metrics['tpr_at_5fpr'] = -1.0

        # 已知类分类准确率（仅对已知类样本）
        known_pred_classes = []
        known_true_filtered = []
        for p, t in zip(known_preds, true_known_labels):
            if not p['is_unknown']:
                known_pred_classes.append(p['predicted_class'])
                known_true_filtered.append(int(t))

        if known_pred_classes:
            metrics['known_accuracy'] = round(
                accuracy_score(known_true_filtered, known_pred_classes), 4)
            metrics['known_f1_macro'] = round(
                f1_score(known_true_filtered, known_pred_classes, average='macro'), 4)
        else:
            metrics['known_accuracy'] = 0.0
            metrics['known_f1_macro'] = 0.0

        # 未知类召回率: 系统正确标记为 unknown 的比例
        unk_truth = np.array([1] * len(unknown_preds))
        unk_pred = np.array([1 if p['is_unknown'] else 0 for p in unknown_preds])
        metrics['unknown_recall'] = round(float(unk_pred.mean()), 4)

        # 已知类误报为 unknown 的比例
        known_false_unknown = sum(1 for p in known_preds if p['is_unknown'])
        metrics['known_false_unknown_rate'] = round(
            known_false_unknown / max(len(known_preds), 1), 4)

        # 综合 F1 (open-set)
        try:
            metrics['f1_open_set'] = round(float(f1_score(y_true, y_pred)), 4)
        except Exception:
            metrics['f1_open_set'] = 0.0

        # RAG 指标 (仅当 use_rag=True)
        if config.use_rag:
            rag_top1_correct = 0
            rag_any_correct = 0
            rag_sims = []
            for p, t in zip(known_preds, true_known_labels):
                rag_res = p.get('rag_result')
                if rag_res:
                    sims = [r['similarity'] for r in rag_res]
                    rag_sims.extend(sims)
                    names = [r['name'] for r in rag_res]
                    true_name = self.label_names[int(t)]
                    if names and names[0] == true_name:
                        rag_top1_correct += 1
                    if true_name in names:
                        rag_any_correct += 1

            metrics['rag_top1_accuracy'] = round(
                rag_top1_correct / max(len(known_preds), 1), 4)
            metrics['rag_top3_accuracy'] = round(
                rag_any_correct / max(len(known_preds), 1), 4)
            metrics['rag_avg_similarity'] = round(
                float(np.mean(rag_sims)) if rag_sims else 0.0, 4)

            # 未知类上的 RAG 相似度（应显著低于已知类）
            unk_rag_sims = []
            for p in unknown_preds:
                rag_res = p.get('rag_result')
                if rag_res:
                    unk_rag_sims.append(rag_res[0]['similarity'])
            metrics['rag_unknown_avg_similarity'] = round(
                float(np.mean(unk_rag_sims)) if unk_rag_sims else 0.0, 4)

        return metrics

    # ───────────────────────── 全量运行 ──────────────────────

    def run_all(self, max_samples: int = 0):
        """运行所有预定义实验并输出报告."""
        print(f"\n{'#'*60}")
        print(f"# Spider-Sense v2 — 消融实验")
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# 数据: processed_cicids (val_known + open_set)")
        print(f"{'#'*60}\n")

        # 加载一次数据
        self.load_data(max_samples)
        self.load_engine()

        results = []
        for cfg in EXPERIMENTS:
            # 反馈实验需要跑在最后（会修改 RAG 状态）
            if cfg.use_rag_feedback:
                continue
            result = self.run_experiment(cfg)
            results.append(result)
            self._print_result(result)

        # 反馈实验 — 用单独的 RAG 实例（避免污染其他实验）
        fb_cfg = [c for c in EXPERIMENTS if c.use_rag_feedback]
        if fb_cfg:
            # 单独加载 RAG
            from backend.rag_engine import RAGEngine
            fb_rag = RAGEngine(knowledge_dir=KNOWLEDGE_DIR)
            fb_rag.build_from_detector(
                detector_path=os.path.join(MODEL_DIR, 'detector.pkl'),
                label_names=self.label_names,
            )
            fb_rag.load_knowledge_base()

            # 临时替换 RAG
            orig_rag = self.rag
            self.rag = fb_rag

            result = self.run_experiment(fb_cfg[0])
            results.append(result)
            self._print_result(result)

            self.rag = orig_rag

        # ── 汇总表格 ──
        self._print_summary(results)

        # ── 保存结果 ──
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(OUTPUT_DIR, f'eval_{timestamp}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[结果] 已保存: {out_path}")

        # 同时保存一份 latest
        latest_path = os.path.join(OUTPUT_DIR, 'latest.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[结果] 已保存: {latest_path}")

        return results

    # ───────────────────────── 打印 ──────────────────────

    def _print_result(self, result: Dict):
        m = result['metrics']
        cfg = result['config']
        print(f"\n  ┌─ {cfg['name']} ─────────────────────────────")
        print(f"  │ AUROC:           {m.get('auroc', 'N/A')}")
        print(f"  │ AUPR(out):       {m.get('aupr_out', 'N/A')}")
        print(f"  │ FPR@95%TPR:      {m.get('fpr_at_95tpr', 'N/A')}")
        print(f"  │ TPR@5%FPR:       {m.get('tpr_at_5fpr', 'N/A')}")
        print(f"  │ Known Accuracy:  {m.get('known_accuracy', 'N/A')}")
        print(f"  │ Known F1(macro): {m.get('known_f1_macro', 'N/A')}")
        print(f"  │ Unknown Recall:  {m.get('unknown_recall', 'N/A')}")
        if 'rag_top1_accuracy' in m:
            print(f"  │ RAG Top-1 Acc:   {m['rag_top1_accuracy']}")
            print(f"  │ RAG Avg Sim:     {m['rag_avg_similarity']}")
        if 'feedback_added' in m:
            print(f"  │ Feedback added:  {m['feedback_added']} patterns")
        print(f"  └{'─'*45}")

    def _print_summary(self, results: List[Dict]):
        """打印汇总对比表."""
        print(f"\n{'='*65}")
        print(f"  消融实验汇总")
        print(f"{'='*65}")
        header = f"  {'Experiment':<30} {'AUROC':<8} {'FPR@95':<8} {'KnownAcc':<8} {'UnkRec':<8}"
        print(header)
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            m = r['metrics']
            name = r['config']['name']
            print(f"  {name:<30} {m.get('auroc', 0):<8.4f} "
                  f"{m.get('fpr_at_95tpr', 0):<8.4f} "
                  f"{m.get('known_accuracy', 0):<8.4f} "
                  f"{m.get('unknown_recall', 0):<8.4f}")
        print(f"{'='*65}")


# ── 独立入口 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Spider-Sense v2 消融实验')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='每类最多样本数 (0=全量, 默认2000)')
    args = parser.parse_args()

    runner = EvalRunner()
    runner.run_all(max_samples=args.max_samples)
