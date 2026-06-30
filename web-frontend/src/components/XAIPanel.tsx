import React, { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface FeatureImportance {
  index: number;
  name: string;
  value: number;
  importance: number;
}

interface XAIResult {
  prediction: string;
  confidence: number;
  is_unknown: boolean;
  feature_importance: FeatureImportance[];
  top_features: FeatureImportance[];
  explanation: string;
  method: string;
}

const XAIPanel: React.FC = () => {
  const [features, setFeatures] = useState('');
  const [result, setResult] = useState<XAIResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExplain = async () => {
    setLoading(true);
    setError(null);
    try {
      const featArray = features.split(',').map(f => parseFloat(f.trim())).filter(f => !isNaN(f));
      const resp = await axios.post('/api/xai/explain', {
        features: featArray,
      });

      if (resp.data?.success) {
        setResult(resp.data.data);
      } else {
        setError(resp.data?.error || '解释失败');
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || '请求失败');
    } finally {
      setLoading(false);
    }
  };

  const handleUseSample = (type: 'ddos' | 'benign') => {
    if (type === 'ddos') {
      setFeatures(Array.from({ length: 64 }, () => (Math.random() * 0.8 + 0.4).toFixed(3)).join(','));
    } else {
      setFeatures(Array.from({ length: 64 }, () => (Math.random() * 0.3).toFixed(3)).join(','));
    }
  };

  // Chart data: top 10 features
  const chartData = result?.feature_importance?.slice(0, 10).map(f => ({
    name: f.name,
    importance: parseFloat((f.importance * 100).toFixed(2)),
  })) || [];

  const topFeatures = result?.top_features?.slice(0, 8) || [];

  return (
    <div>
      <h2 className="panel-title">
        <span>📊</span> XAI Explainable Analysis
      </h2>
      <p style={{ color: '#888', marginBottom: 20 }}>
        基于特征扰动的归因分析，解释模型为何做出特定判定
      </p>

      {/* Input */}
      <div style={{
        padding: '16px 18px',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 10,
        marginBottom: 20,
      }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>特征向量</div>
          <textarea
            value={features}
            onChange={e => setFeatures(e.target.value)}
            placeholder="输入特征向量，用逗号分隔..."
            rows={2}
            style={{
              width: '100%',
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 6,
              padding: '8px 12px',
              color: '#e0e0e0',
              fontFamily: 'monospace',
              fontSize: 12,
              resize: 'vertical',
            }}
          />
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button
            className="btn btn-primary"
            onClick={handleExplain}
            disabled={loading || !features}
          >
            {loading ? '🔄 分析中...' : '🔬 生成XAI解释'}
          </button>
          <button className="btn btn-secondary" onClick={() => handleUseSample('ddos')} style={{ fontSize: 12 }}>
            💥 攻击样本
          </button>
          <button className="btn btn-secondary" onClick={() => handleUseSample('benign')} style={{ fontSize: 12 }}>
            ✅ 正常样本
          </button>
        </div>
      </div>

      {error && (
        <div style={{ color: '#ff4757', marginBottom: 16, padding: '10px 14px', background: 'rgba(255,71,87,0.1)', borderRadius: 8 }}>
          ⚠ {error}
        </div>
      )}

      {result && (
        <div>
          {/* Prediction Banner */}
          <div style={{
            padding: '14px 18px',
            background: result.is_unknown
              ? 'rgba(255,71,87,0.1)' : 'rgba(46,213,115,0.1)',
            border: `2px solid ${result.is_unknown ? '#ff4757' : '#2ed573'}`,
            borderRadius: 10,
            marginBottom: 20,
          }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#fff', marginBottom: 6 }}>
              {result.is_unknown ? '⚠️ 未知攻击检测' : `✅ 已知类别: ${result.prediction}`}
              <span style={{ fontSize: 13, color: '#aaa', marginLeft: 12 }}>
                置信度: {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ color: '#ddd', fontSize: 13, lineHeight: 1.6 }}>
              {result.explanation}
            </div>
            <div style={{ marginTop: 6, color: '#666', fontSize: 11 }}>
              分析方法: {result.method}
            </div>
          </div>

          {/* Feature Importance Chart */}
          <div style={{
            padding: '16px 18px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 10,
            marginBottom: 20,
          }}>
            <h4 style={{ color: '#888', marginTop: 0, marginBottom: 16 }}>Top 10 特征重要性</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" tick={{ fill: '#888' }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#aaa', fontSize: 11 }} width={140} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: number) => [`${value}%`, '重要性']}
                />
                <Bar dataKey="importance" fill="#8a2be2" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top Features Detail */}
          <div style={{
            padding: '16px 18px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 10,
          }}>
            <h4 style={{ color: '#888', marginTop: 0, marginBottom: 16 }}>Top 特征归因详情</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
              {topFeatures.map((f, i) => (
                <div key={f.index} style={{
                  padding: '10px 14px',
                  background: 'rgba(0,0,0,0.2)',
                  borderRadius: 8,
                  borderLeft: `3px solid hsl(${280 - i * 30}, 70%, 50%)`,
                }}>
                  <div style={{ color: '#e0e0e0', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
                    {i + 1}. {f.name}
                  </div>
                  <div style={{ color: '#666', fontSize: 11, marginBottom: 4 }}>
                    原始值: {f.value.toFixed(4)}
                  </div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                  }}>
                    <div style={{
                      flex: 1,
                      height: 4,
                      background: 'rgba(255,255,255,0.1)',
                      borderRadius: 2,
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${Math.min(100, f.importance * 500)}%`,
                        height: '100%',
                        background: `hsl(${280 - i * 30}, 70%, 50%)`,
                        borderRadius: 2,
                      }} />
                    </div>
                    <span style={{ color: '#aaa', fontSize: 10 }}>
                      {(f.importance * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div style={{ textAlign: 'center', padding: 60, color: '#666' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
          <div style={{ fontSize: 16, marginBottom: 8 }}>输入特征向量生成XAI解释</div>
          <div style={{ fontSize: 13 }}>
            通过特征扰动分析，揭示哪些特征对判定结果贡献最大
          </div>
        </div>
      )}
    </div>
  );
};

export default XAIPanel;
