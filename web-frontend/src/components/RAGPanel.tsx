import React, { useState } from 'react';
import axios from 'axios';

interface RAGMatch {
  pattern_id: string;
  similarity: number;
  name: string;
  description: string;
  severity: string;
  category: string;
  mitre: {
    id: string;
    name: string;
    tactics: string[];
    description: string;
  } | null;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: '#ff4757',
  high: '#ff6348',
  medium: '#ffc107',
  low: '#2ed573',
};

const RAGPanel: React.FC = () => {
  const [features, setFeatures] = useState('');
  const [results, setResults] = useState<RAGMatch[]>([]);
  const [historyMatches, setHistoryMatches] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const featArray = features.split(',').map(f => parseFloat(f.trim())).filter(f => !isNaN(f));
      const resp = await axios.post('/api/rag/search', {
        features: featArray,
        top_k: 5,
      });

      if (resp.data?.success) {
        setResults(resp.data.data.matches || []);
        setHistoryMatches(resp.data.data.history_matches || []);
        setStats(resp.data.data.stats);
      } else {
        setError(resp.data?.error || '检索失败');
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || '请求失败');
    } finally {
      setLoading(false);
    }
  };

  const handleUseSample = () => {
    setFeatures(Array.from({ length: 64 }, () => (Math.random() * 0.8 + 0.2).toFixed(3)).join(','));
  };

  return (
    <div>
      <h2 className="panel-title">
        <span>🔍</span> RAG Knowledge Base Search
      </h2>
      <p style={{ color: '#888', marginBottom: 20 }}>
        基于向量相似度检索攻击模式库和MITRE ATT&CK知识图谱
      </p>

      {/* Search */}
      <div style={{
        padding: '16px 18px',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 10,
        marginBottom: 20,
      }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>特征向量或嵌入向量</div>
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
            onClick={handleSearch}
            disabled={loading || !features}
          >
            {loading ? '🔄 检索中...' : '🔍 搜索相似攻击模式'}
          </button>
          <button className="btn btn-secondary" onClick={handleUseSample} style={{ fontSize: 13 }}>
            🎲 随机样本
          </button>
        </div>
      </div>

      {/* Knowledge Stats */}
      {stats && (
        <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3,1fr)', marginBottom: 20 }}>
          <div className="metric-card">
            <div className="metric-value" style={{ color: '#00d4ff' }}>{stats.num_patterns}</div>
            <div className="metric-label">攻击模式</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{ color: '#ffc107' }}>{stats.num_mitre_techniques}</div>
            <div className="metric-label">MITRE技术</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{ color: '#2ed573' }}>{stats.num_history_entries}</div>
            <div className="metric-label">历史记录</div>
          </div>
        </div>
      )}

      {error && (
        <div style={{ color: '#ff4757', marginBottom: 16, padding: '10px 14px', background: 'rgba(255,71,87,0.1)', borderRadius: 8 }}>
          ⚠ {error}
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && (
        <div>
          <h3 style={{ color: '#fff', marginBottom: 16 }}>匹配结果</h3>
          {results.map((match, i) => (
            <div key={match.pattern_id} style={{
              padding: '16px 18px',
              background: i === 0 ? 'rgba(0,212,255,0.05)' : 'rgba(255,255,255,0.03)',
              border: i === 0 ? '1px solid rgba(0,212,255,0.2)' : '1px solid rgba(255,255,255,0.06)',
              borderRadius: 10,
              marginBottom: 12,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ color: '#fff', fontWeight: 600, fontSize: 15 }}>
                    {match.name}
                  </div>
                  {match.mitre && (
                    <div style={{ color: '#00d4ff', fontSize: 12, marginTop: 4 }}>
                      {match.mitre.id}: {match.mitre.name}
                    </div>
                  )}
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    display: 'inline-block',
                    padding: '3px 10px',
                    borderRadius: 12,
                    fontSize: 11,
                    fontWeight: 700,
                    background: `${SEVERITY_COLORS[match.severity] || '#666'}22`,
                    color: SEVERITY_COLORS[match.severity] || '#888',
                    border: `1px solid ${SEVERITY_COLORS[match.severity] || '#666'}44`,
                  }}>
                    {match.severity.toUpperCase()}
                  </div>
                  <div style={{ color: '#00d4ff', fontSize: 18, fontWeight: 700, marginTop: 8 }}>
                    {(match.similarity * 100).toFixed(1)}%
                  </div>
                  <div style={{ color: '#666', fontSize: 10 }}>相似度</div>
                </div>
              </div>

              <div style={{ color: '#aaa', fontSize: 12, marginTop: 10, lineHeight: 1.6 }}>
                {match.description}
              </div>

              {/* MITRE details */}
              {match.mitre && (
                <div style={{
                  marginTop: 12,
                  padding: '10px 14px',
                  background: 'rgba(0,0,0,0.2)',
                  borderRadius: 6,
                }}>
                  <div style={{ color: '#ffc107', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
                    MITRE ATT&CK: {match.mitre.id}
                  </div>
                  <div style={{ color: '#aaa', fontSize: 11, marginBottom: 4 }}>{match.mitre.description}</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    <span style={{ color: '#888', fontSize: 10 }}>Tactics:</span>
                    {match.mitre.tactics.map(t => (
                      <span key={t} style={{
                        padding: '2px 8px',
                        background: 'rgba(255,193,7,0.1)',
                        borderRadius: 10,
                        color: '#ffc107',
                        fontSize: 10,
                      }}>
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Category tag */}
              <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
                {match.category.split(',').map(c => (
                  <span key={c} style={{
                    padding: '2px 8px',
                    background: 'rgba(138,43,226,0.1)',
                    borderRadius: 10,
                    color: '#c38aff',
                    fontSize: 10,
                  }}>
                    {c.trim()}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {results.length === 0 && !loading && (
        <div style={{ textAlign: 'center', padding: 60, color: '#666' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
          <div style={{ fontSize: 16, marginBottom: 8 }}>输入特征向量开始检索</div>
          <div style={{ fontSize: 13 }}>
            RAG引擎将在攻击模式库和MITRE ATT&CK知识图谱中搜索最相似的攻击模式
          </div>
        </div>
      )}
    </div>
  );
};

export default RAGPanel;
