import React, { useState } from 'react';

interface Props { event: any; data: any; loading: boolean; onAnalyze?: () => void; }

const VERDICT_CN: Record<string, string> = {
  CONFIRMED_THREAT: '确认威胁', LIKELY_THREAT: '疑似威胁',
  UNCERTAIN: '不确定', LIKELY_BENIGN: '疑似误报',
};

const DetailDrawer: React.FC<Props> = ({ event, data, loading, onAnalyze }) => {
  const [tab, setTab] = useState<'agent' | 'rag' | 'xai'>('agent');

  if (loading) return <div style={{ color: 'var(--muted)', padding: 24, textAlign: 'center', fontSize: 13 }}>正在加载分析数据...</div>;
  if (!data) return <div style={{ color: 'var(--muted)', padding: 24, textAlign: 'center', fontSize: 13 }}>暂无分析数据</div>;

  const { debate, rag, xai } = data;

  const sectionHeader = { fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 8 };
  const sectionBody = { fontSize: 12, color: 'var(--text)', lineHeight: 1.8, whiteSpace: 'pre-wrap' as const };

  return (
    <div>
      <div className="drawer-tabs">
        <button className={`drawer-tab ${tab === 'agent' ? 'active' : ''}`} onClick={() => setTab('agent')}>🛡️ 智能研判</button>
        <button className={`drawer-tab ${tab === 'rag' ? 'active' : ''}`} onClick={() => setTab('rag')}>🔍 知识库证据</button>
        <button className={`drawer-tab ${tab === 'xai' ? 'active' : ''}`} onClick={() => setTab('xai')}>📊 决策归因</button>
      </div>

      {/* Agent 研判报告 — 单安全专家智能体（按需触发） */}
      {tab === 'agent' && (
        <div>
          {!debate ? (
            <div style={{ textAlign: 'center', padding: '40px 20px' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>🛡️</div>
              <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 16 }}>
                Senior SecOps 安全专家智能体
                <br />融合 XAI 特征归因 + RAG 威胁情报，生成结构化研判报告。
              </div>
              <button
                onClick={onAnalyze}
                disabled={loading}
                style={{
                  padding: '10px 24px',
                  borderRadius: 8,
                  border: '1px solid var(--border)',
                  background: 'var(--surface2)',
                  color: 'var(--text)',
                  cursor: 'pointer',
                  fontSize: 13,
                }}
              >
                {loading ? '⏳ 分析中...' : '🛡️ 运行智能分析'}
              </button>
            </div>
          ) : (
            <>
              {/* 判定结果横幅 */}
              <div className="final-verdict" style={{ marginBottom: 16 }}>
                <h4>
                  {debate.verdict === 'MALICIOUS' ? '🚨 恶意流量' : debate.verdict === 'SUSPICIOUS' ? '⚠️ 可疑流量' : '✅ 良性 / 误报'}
                  <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 400, marginLeft: 8 }}>
                    威胁评分: {(debate.threat_score * 100).toFixed(0)}%
                  </span>
                </h4>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                  分析模式: {debate.analysis_mode === 'llm' ? `LLM (${debate.llm_model || 'N/A'})` : '规则引擎（离线模板）'}
                </div>
              </div>

              {/* 1. 风险评估 */}
              {debate.risk_assessment && (
                <div style={{ marginBottom: 16 }}>
                  <div style={sectionHeader}>🔍 风险评估</div>
                  <div style={sectionBody}>{debate.risk_assessment}</div>
                </div>
              )}

              {/* 2. 情报关联 */}
              {debate.intelligence_correlation && (
                <div style={{ marginBottom: 16 }}>
                  <div style={sectionHeader}>🎯 情报关联</div>
                  <div style={sectionBody}>{debate.intelligence_correlation}</div>
                </div>
              )}

              {/* 3. 处置建议 */}
              {debate.disposal_recommendation && (
                <div style={{ marginBottom: 16 }}>
                  <div style={sectionHeader}>🛡️ 处置建议</div>
                  <div style={sectionBody}>{debate.disposal_recommendation}</div>
                </div>
              )}

              {/* 兼容旧版三智能体格式 (fallback) */}
              {debate.detector && !debate.risk_assessment && (
                <div className="agent-grid">
                  {[
                    { key: 'detector', name: '检测员', icon: '🕵️', data: debate.detector },
                    { key: 'analyst', name: '分析师', icon: '🔬', data: debate.analyst },
                    { key: 'arbiter', name: '裁决官', icon: '⚖️', data: debate.arbiter },
                  ].map(a => (
                    <div key={a.key} className="agent-card">
                      <div className="agent-header">
                        <div>
                          <div className="agent-name">{a.icon} {a.name}</div>
                        </div>
                        <span className={`tag ${a.data?.verdict?.includes('THREAT') ? 'tag-red' : a.data?.verdict?.includes('BENIGN') ? 'tag-green' : 'tag-yellow'}`}>
                          {VERDICT_CN[a.data?.verdict] || a.data?.verdict || 'N/A'}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>
                        {a.data?.analysis || ''}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* RAG Evidence */}
      {tab === 'rag' && rag && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 12 }}>
            基于当前流量特征向量，在攻击模式知识库中检索最相似的已知攻击模式：
          </div>
          <div className="rag-list">
            {(rag.matches || []).length > 0 ? rag.matches.map((m: any, i: number) => (
              <div key={i} className="rag-item">
                <div className="rag-header">
                  <div className="rag-name">{m.name}</div>
                  <div className="rag-sim">相似度 {(m.similarity * 100).toFixed(0)}%</div>
                </div>
                <div className="rag-desc">{m.description}</div>
                {m.mitre && (
                  <div className="rag-mitre">
                    <span className="tag tag-purple">{m.mitre.id}</span>
                    <span style={{ marginLeft: 6, color: 'var(--muted)' }}>{m.mitre.name}</span>
                    {m.mitre.tactics?.length > 0 && (
                      <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--muted)' }}>
                        战术阶段: {m.mitre.tactics.join(', ')}
                      </span>
                    )}
                  </div>
                )}
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                  严重程度: <span style={{ color: m.severity === 'critical' ? 'var(--red)' : m.severity === 'high' ? 'var(--yellow)' : 'var(--green)' }}>
                    {m.severity === 'critical' ? '严重' : m.severity === 'high' ? '高' : m.severity === 'medium' ? '中' : '低'}
                  </span>
                </div>
              </div>
            )) : <div style={{ color: 'var(--muted)', fontSize: 12 }}>知识库中未找到相似攻击模式</div>}
          </div>
          {rag.stats && (
            <div style={{ marginTop: 16, fontSize: 10, color: 'var(--muted)' }}>
              知识库规模: {rag.stats.semantic_rag?.num_entries || rag.stats.knowledge?.num_mitre_techniques || 0} 项 MITRE ATT&CK 技术
              {rag.stats.semantic_rag?.embedding_dim && <span> · {rag.stats.semantic_rag.embedding_dim} 维语义向量</span>}
              {rag.stats.knowledge?.num_history_entries > 0 && <span> · {rag.stats.knowledge.num_history_entries} 条历史记录</span>}
            </div>
          )}
        </div>
      )}

      {/* XAI */}
      {tab === 'xai' && xai && (
        <div>
          <div style={{ fontSize: 13, color: 'var(--text)', marginBottom: 16, lineHeight: 1.6, padding: '10px 12px', background: 'var(--bg)', borderRadius: 6, border: '1px solid var(--border)' }}>
            {xai.explanation || '无法生成解释'}
          </div>

          <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 10, fontWeight: 600 }}>特征贡献排序（扰动归因分析）</div>
          {(xai.top_features || []).length > 0 ? (
            <div className="xai-features">
              {xai.top_features.slice(0, 10).map((f: any, i: number) => (
                <div key={i} className="xai-feature">
                  <div className="f-name" title={f.name}>{f.name}</div>
                  <div className="f-bar">
                    <div className="f-fill" style={{
                      width: `${Math.min(100, f.importance * 200)}%`,
                      background: f.importance > 0.005 ? 'var(--red)' : 'var(--blue)',
                    }} />
                  </div>
                  <div className="f-val">{(f.importance * 100).toFixed(2)}%</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--muted)', fontSize: 12 }}>该事件的模型置信度过高，特征扰动未产生显著影响，无法提取有效的特征归因。</div>
          )}

          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 16 }}>
            分析方法: {xai.method || '特征扰动'} &middot; 判定结果: {xai.prediction} &middot; 未知攻击: {xai.is_unknown ? '是' : '否'}
          </div>
        </div>
      )}
    </div>
  );
};

export default DetailDrawer;
